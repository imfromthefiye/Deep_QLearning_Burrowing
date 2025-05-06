import pygame
import numpy as np
import time
import gymnasium as gym
from gymnasium import spaces
from Box2D import (
    b2World, b2PolygonShape, b2FixtureDef, b2CircleShape,
    b2PrismaticJointDef, b2ContactListener, b2ContactImpulse,
    b2MouseJointDef, b2RevoluteJointDef, b2WeldJointDef
)
import cv2  # Add OpenCV for video recording
import os

# ================== Environment Constants ==================
CHAMBER_WIDTH = 0.8   # meters
CHAMBER_HEIGHT = 1.0  # meters
MARGIN = 0.1          # meters border
SCALE = 400.0         # px per meter
VIEWPORT_W = int((CHAMBER_WIDTH + 2 * MARGIN) * SCALE)
VIEWPORT_H = int((CHAMBER_HEIGHT + 2 * MARGIN) * SCALE)
FPS = 60
TIME_STEP = 1.0 / FPS
VEL_ITERS, POS_ITERS = 10, 10

# Grain parameters
GRAIN_RADIUS = 0.015  # m

# Probe parameters
SHAFT_W, SHAFT_H = 0.05, 0.5
TIP_BASE = SHAFT_W
TIP_HEIGHT = SHAFT_W * 0.6

# Position the probe closer to the soil bed
soil_height = CHAMBER_HEIGHT / 3
PROBE_POS_Y = soil_height + 0.05 + SHAFT_H/2  # Only small clearance above soil

# Penetration parameters
PENETRATION_TARGET = 0.8  # Target depth in meters
FULL_SPEED = 0.05      # m/s (maximum penetration speed)
HALF_SPEED = 0.025     # m/s (half of maximum speed)
MAX_SAFE_SPEED = 0.06  # m/s (speed safety threshold)
MAX_FORCE = 100.0      # Maximum force before normalization
MAX_STEPS = 1000       # Maximum number of steps before termination

# ================== Helper: world->screen ==================
def world_to_screen(x, y):
    # Move the origin to bottom-left corner of the chamber
    # This positions the chamber floor at the bottom of the viewport with a small margin
    sx = int((x + CHAMBER_WIDTH/2 + MARGIN) * SCALE)
    
    # Flip Y-axis and position the origin at the bottom with a small margin
    # This ensures the floor appears at the bottom of the screen
    bottom_margin = MARGIN * 2  # Extra margin at the bottom
    sy = int(VIEWPORT_H - ((y + bottom_margin) * SCALE))
    
    return sx, sy

def world_to_screen_pts(pt):
    x, y = pt
    return world_to_screen(x, y)

# ================== Contact Listener ==================
class ContactDetector(b2ContactListener):
    def __init__(self):
        super().__init__()
        self.tip_impulse = 0.0

    def PostSolve(self, contact, impulse: b2ContactImpulse):
        total = sum(impulse.normalImpulses)
        self.tip_impulse += total

# ================== ConePen Environment ==================
class ConePenEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}
    
    def __init__(self, render_mode=None):
        # Initialize rendering components
        self.screen = None
        self.clock = None
        self.isopen = True
        self.render_mode = render_mode
        
        # Initialize physics world
        self.world = None
        self.chamber = None
        self.grains = []
        self.shaft = None
        self.tip = None
        self.joint = None
        
        # State tracking
        self.initial_probe_y = None
        self.current_depth = 0.0
        self.current_velocity = 0.0
        self.current_force = 0.0
        self.step_count = 0
        self.game_over = False
        self.contact_detector = None
        
        # Define observation space (normalized values)
        # [depth_ratio, tip_speed_norm, tip_force_norm, time_frac]
        self.observation_space = spaces.Box(
            low=np.array([-1.2, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Define action space
        # 0: Retract, 1: Penetrate fast, 2: Stop
        self.action_space = spaces.Discrete(3)
    
    def _destroy(self):
        """Clean up Box2D objects"""
        if self.world is None:
            return
            
        # Destroy grains
        for grain in self.grains:
            self.world.DestroyBody(grain)
        self.grains = []
        
        # Destroy probe parts
        if self.shaft is not None:
            self.world.DestroyBody(self.shaft)
            self.shaft = None
        if self.tip is not None:
            self.world.DestroyBody(self.tip)
            self.tip = None
        
        # Destroy chamber
        if self.chamber is not None:
            for wall in self.chamber:
                self.world.DestroyBody(wall)
            self.chamber = None
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Clean up existing objects
        self._destroy()
        
        # Create new world
        self.world = b2World(gravity=(0, -9.81))
        self.game_over = False
        self.step_count = 0
        
        # Create contact detector
        self.contact_detector = ContactDetector()
        self.world.contactListener = self.contact_detector
        
        # Create chamber walls
        self.chamber = self._create_chamber()
        
        # Create soil grains
        self.grains = self._create_grains()
        
        # Create probe (shaft and tip as separate bodies)
        self.shaft, self.tip = self._create_probe()
        
        # Store initial probe position
        self.initial_probe_y = self.shaft.position.y
        self.current_depth = 0.0
        self.current_velocity = 0.0
        self.current_force = 0.0
        
        # Let the soil settle for a bit
        for _ in range(100):
            self.world.Step(TIME_STEP, VEL_ITERS, POS_ITERS)
            self.contact_detector.tip_impulse = 0.0
        
        if self.render_mode == "human":
            self.render()
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Take a step in the environment"""
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        self.step_count += 1
        
        # Apply action to control penetration speed
        if action == 0:    # Retract (upward half-speed)
            self.joint.motorSpeed = +HALF_SPEED
        elif action == 1:  # Penetrate fast (down at full speed)
            self.joint.motorSpeed = -FULL_SPEED
        elif action == 2:  # Stop
            self.joint.motorSpeed = 0.0
        
        # Reset force measurement before physics step
        self.contact_detector.tip_impulse = 0.0
        
        # Step physics
        self.world.Step(TIME_STEP, VEL_ITERS, POS_ITERS)
        
        # Update tip position to follow shaft
        self.tip.position = (0.0, self.shaft.position.y - SHAFT_H/2 - TIP_HEIGHT/2)
        
        # Update state variables
        self._update_state()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        # success
        if self.current_depth >= 1.0:
            terminated = True
        # failure
        elif self.current_depth < -1.1:
            terminated = True
        # safety (excess speed)
        elif abs(self.current_velocity) > MAX_SAFE_SPEED / FULL_SPEED:
            terminated = True
        
        # truncation
        if self.step_count >= MAX_STEPS:
            truncated = True
        
        if self.render_mode == "human":
            self.render()
        
        return self._get_observation(), reward, terminated, truncated, {}
    
    def _update_state(self):
        """Update the current state variables"""
        # a) depth ratio
        raw_depth = self.initial_probe_y - self.shaft.position.y
        self.current_depth = raw_depth / PENETRATION_TARGET
        
        # b) normalized tip velocity
        self.current_velocity = (self.shaft.linearVelocity.y / FULL_SPEED)
        
        # c) normalized contact force
        self.current_force = np.clip(self.contact_detector.tip_impulse / MAX_FORCE, 0.0, 1.0)
    
    def _calculate_reward(self):
        """Calculate reward based on current state"""
        # 1) Progress reward
        prev = getattr(self, 'prev_depth', 0.0)
        progress = max(0.0, self.current_depth - prev)
        reward = progress * 2.0

        # 2) Force penalty
        reward -= (self.current_force ** 2)

        # 3) Time penalty
        reward -= 0.01

        # 4) Terminal bonuses/penalties
        if self.current_depth >= 1.0:
            reward += 10.0    # success
        elif self.current_depth < -1.1:
            reward -= 10.0    # over-penetration

        self.prev_depth = self.current_depth
        return reward
    
    def _get_observation(self):
        """Get the current state observation"""
        depth_ratio = np.clip(self.current_depth, -1.2, 0.0)
        time_frac = self.step_count / MAX_STEPS
        return np.array([
            depth_ratio,
            self.current_velocity,
            self.current_force,
            time_frac
        ], dtype=np.float32)
    
    def _create_chamber(self):
        """Create the chamber walls"""
        # Wall thickness
        t = 0.01
        
        # Calculate half width
        half_w = CHAMBER_WIDTH / 2
        
        walls = []
        
        # Left wall - place it just outside the leftmost grains
        walls.append(self.world.CreateStaticBody(
            position=(-half_w - t/2, CHAMBER_HEIGHT/2),
            fixtures=b2FixtureDef(shape=b2PolygonShape(box=(t/2, CHAMBER_HEIGHT/2)), friction=0.6)
        ))
        
        # Right wall - place it just outside the rightmost grains
        walls.append(self.world.CreateStaticBody(
            position=(half_w + t/2, CHAMBER_HEIGHT/2),
            fixtures=b2FixtureDef(shape=b2PolygonShape(box=(t/2, CHAMBER_HEIGHT/2)), friction=0.6)
        ))
        
        # Floor - place it directly beneath the grains at y=0
        walls.append(self.world.CreateStaticBody(
            position=(0.0, -t/2),
            fixtures=b2FixtureDef(shape=b2PolygonShape(box=(half_w + t, t/2)), friction=0.6)
        ))
        
        return walls
    
    def _create_grains(self):
        """Create soil grains in a hexagonal packing structure"""
        grains = []
        
        # Define the grain arrangement first
        grain_diameter = GRAIN_RADIUS * 2
        
        # Calculate rows and columns based on chamber width, not including walls yet
        COLS = int(CHAMBER_WIDTH / grain_diameter) + 2  # Add extra columns to ensure full coverage
        
        # Fill 1/3 of chamber height with grains
        soil_height = CHAMBER_HEIGHT / 3
        ROWS = int(soil_height / (grain_diameter * 0.84))  # Reduced from 0.866 for tighter packing
        
        # Center the array horizontally
        x0 = -(COLS-1) * grain_diameter / 2
        
        # Position grains starting at y=0 (we'll adjust the chamber floor later)
        y0 = GRAIN_RADIUS  # Bottom of first grain at y=0
        
        for i in range(ROWS):
            for j in range(COLS):
                # Offset alternate rows for hexagonal packing
                x = x0 + j * grain_diameter + (0.5 * grain_diameter if i % 2 else 0)
                y = y0 + i * (grain_diameter * 0.84)  # Reduced from 0.866 for tighter packing
                
                # Add random jitter to avoid perfect lattice arrangement
                x += np.random.uniform(-0.002, 0.002)
                y += np.random.uniform(-0.002, 0.002)
                
                # Allow grains to extend closer to chamber walls (only skip if majority of grain would be outside)
                if abs(x) > (CHAMBER_WIDTH/2 + GRAIN_RADIUS/2):
                    continue
                    
                body = self.world.CreateDynamicBody(
                    position=(x, y),
                    fixtures=b2FixtureDef(
                        shape=b2CircleShape(radius=GRAIN_RADIUS),
                        density=2.5,       # Increased mass to resist movement
                        friction=0.6,      # friction for resistance
                        restitution=0.0    # no bounce
                    ),
                    linearDamping=0.5,   # Reduced from 0.8 to allow more flow
                    angularDamping=0.6    # Reduced from 0.9 to allow more rotation
                )
                # Enable continuous collision detection to prevent tunneling
                body.bullet = True
                grains.append(body)
        
        return grains
    
    def _create_probe(self):
        """Create single probe (shaft and tip as separate bodies)"""
        # Create shaft 
        self.shaft = self.world.CreateDynamicBody(
            position=(0.0, PROBE_POS_Y),
            fixedRotation=True,
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(box=(SHAFT_W/2, SHAFT_H/2)),
                density=1.0, friction=0.2
            )
        )
        
        # Create tip
        self.tip = self.world.CreateDynamicBody(
            position=(0.0, PROBE_POS_Y - SHAFT_H/2 - TIP_HEIGHT/2),
            fixedRotation=True,
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(vertices=[
                    (-TIP_BASE/2, TIP_HEIGHT/2),
                    (TIP_BASE/2, TIP_HEIGHT/2),
                    (0.0, -TIP_HEIGHT/2)
                ]),
                density=1.0, friction=0.2
            )
        )
        
        # Disable gravity for probe parts
        self.shaft.gravityScale = 0.0
        self.tip.gravityScale = 0.0
        
        # Create vertical joint for shaft movement
        self.joint = self._create_vertical_joint(self.shaft)
        
        # Store initial position
        self.initial_probe_y = self.shaft.position.y
        
        return self.shaft, self.tip
    
    def _create_vertical_joint(self, body):
        """Create vertical joint for probe movement"""
        ground = self.world.CreateStaticBody(position=(0,0))
        
        joint = b2PrismaticJointDef()
        joint.Initialize(ground, body, anchor=body.position, axis=(0, 1))
        joint.enableLimit = True
        joint.lowerTranslation = -CHAMBER_HEIGHT  # can go down
        joint.upperTranslation = 0.0  # cannot go up past start
        joint.enableMotor = True
        joint.motorSpeed = 0.0
        joint.maxMotorForce = 500.0
        
        return self.world.CreateJoint(joint)
    
    def render(self):
        """Render the environment"""
        if self.render_mode is None:
            return
        
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
            pygame.display.set_caption("Cone Penetration Test")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)  # Default font, size 24
        
        # Colors exactly matching the AnchorExp environment code snippets
        WALL_COLOR = (80, 80, 80)        # Dark gray
        GRAIN_COLOR = (150, 150, 150)    # Medium gray (from snippet)
        SHAFT_COLOR = (100, 100, 255)    # Brighter blue (from snippet)
        TIP_COLOR = (255, 100, 100)      # Brighter red (from snippet)
        TEXT_COLOR = (0, 0, 0)           # Black
        BG_COLOR = (255, 255, 255)       # White
        
        self.screen.fill(BG_COLOR)
        
        # Draw chamber walls
        for wall in self.chamber:
            for fixture in wall.fixtures:
                shape = fixture.shape
                vertices = [(wall.transform * v) for v in shape.vertices]
                vertices = [world_to_screen_pts(v) for v in vertices]
                pygame.draw.polygon(self.screen, WALL_COLOR, vertices)
        
        # Draw soil grains
        for grain in self.grains:
            p = world_to_screen_pts(grain.position)
            radius = int(GRAIN_RADIUS * SCALE)
            pygame.draw.circle(self.screen, GRAIN_COLOR, p, radius)
        
        # Draw shaft
        for fixture in self.shaft.fixtures:
            shape = fixture.shape
            vertices = [(self.shaft.transform * v) for v in shape.vertices]
            vertices = [world_to_screen_pts(v) for v in vertices]
            pygame.draw.polygon(self.screen, SHAFT_COLOR, vertices)
        
        # Draw tip
        for fixture in self.tip.fixtures:
            shape = fixture.shape
            vertices = [(self.tip.transform * v) for v in shape.vertices]
            vertices = [world_to_screen_pts(v) for v in vertices]
            pygame.draw.polygon(self.screen, TIP_COLOR, vertices)
        
        # Display debug information
        lines = [
            f"DepthR: {self.current_depth:.2f}",
            f"Speed : {self.current_velocity:.2f}",
            f"Force : {self.current_force:.2f}",
            f"Time  : {self.step_count/MAX_STEPS:.2f}"
        ]
        
        y_pos = 10
        for line in lines:
            text = self.font.render(line, True, TEXT_COLOR)
            self.screen.blit(text, (10, y_pos))
            y_pos += 25
        
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
    
    def close(self):
        """Close the environment and clean up resources"""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.isopen = False

def main():
    """Run the environment for debugging purposes"""
    env = ConePenEnv(render_mode="human")
    observation, info = env.reset()
    
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Initial observation:", observation)
    
    total_reward = 0
    step_count = 0
    
    try:
        # Simple policy for testing
        while True:
            # If we're above target depth (negative values), penetrate
            # Otherwise stop or retract
            depth = observation[0]
            if depth > -0.9:
                action = 1  # Penetrate
            elif depth < -1.0:
                action = 0  # Retract
            else:
                action = 2  # Stop
                
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step_count += 1
            
            if step_count % 50 == 0:
                print(f"Step {step_count}: depth={observation[0]:.2f}, vel={observation[1]:.2f}, force={observation[2]:.2f}, time={observation[3]:.2f}")
                print(f"  Action={action}, Reward={reward:.2f}, Total={total_reward:.2f}")
            
            done = terminated or truncated
            if done:
                print(f"Episode complete: {step_count} steps, reward: {total_reward:.2f}")
                print(f"Final depth: {observation[0]}")
                break
                
            # Process events to allow window close
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    done = True
            if done:
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        env.close()
        print("Environment closed")

if __name__ == "__main__":
    main()