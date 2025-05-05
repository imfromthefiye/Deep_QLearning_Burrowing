import pygame
from Box2D import (
    b2World, b2PolygonShape, b2FixtureDef, b2CircleShape,
    b2PrismaticJointDef, b2ContactListener, b2ContactImpulse
)
import time
import cv2  # Add OpenCV for video recording
import os
import numpy as np

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

# ================== World Setup ==================
def create_grains(world):
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
                
            body = world.CreateDynamicBody(
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

def create_chamber(world):
    # Wall thickness
    t = 0.01
    
    # Calculate half width
    half_w = CHAMBER_WIDTH / 2
    
    walls = []
    
    # Left wall - place it just outside the leftmost grains
    walls.append(world.CreateStaticBody(
        position=(-half_w - t/2, CHAMBER_HEIGHT/2),
        fixtures=b2FixtureDef(shape=b2PolygonShape(box=(t/2, CHAMBER_HEIGHT/2)), friction=0.6)
    ))
    
    # Right wall - place it just outside the rightmost grains
    walls.append(world.CreateStaticBody(
        position=(half_w + t/2, CHAMBER_HEIGHT/2),
        fixtures=b2FixtureDef(shape=b2PolygonShape(box=(t/2, CHAMBER_HEIGHT/2)), friction=0.6)
    ))
    
    # Floor - place it directly beneath the grains at y=0
    walls.append(world.CreateStaticBody(
        position=(0.0, -t/2),
        fixtures=b2FixtureDef(shape=b2PolygonShape(box=(half_w + t, t/2)), friction=0.6)
    ))
    
    return walls

# ================== Probe Parameters ==================
# Probe parameters
SHAFT_W, SHAFT_H = 0.05, 0.5
TIP_BASE = SHAFT_W
TIP_HEIGHT = SHAFT_W * 0.6

# Position the probe closer to the soil bed
soil_height = CHAMBER_HEIGHT / 3
PROBE_POS_Y = soil_height + 0.05 + SHAFT_H/2  # Only small clearance above soil
EXPANSION_RATIO = 1.2
TIP_EXTENSION_SPEED = 0.03

# Autonomous motion parameters
PENETRATION_SPEED = 0.05      # m/s (slower for better physics)
EXPANSION_SPEED = 0.05        # m/s (horizontal expansion speed)
RETRACTION_SPEED = 0.05       # m/s (tip retraction speed)
BODY_FOLLOW_SPEED = 0.08      # m/s (body follows tip speed)

# Target depths and distances
PENETRATION_TARGET = -0.24    # m (target penetration depth for tip)
EXPANSION_DISTANCE = 0.1      # m (how far to expand each shaft)
TIP_EXTENSION_DISTANCE = 0.05 # m (how far to extend the tip)

# Phase duration limits
SETTLING_TIME = 2.0           # seconds for initial settling
EXPANSION_TIME = 2.0          # seconds for expansion phase
EXTENSION_TIME = 1.0          # seconds for tip extension
MAX_PHASE_TIME = 10.0         # maximum time per phase as safety

# ================== Phases ==================
PHASES = [
    'settling',          # Let the soil settle
    'cone_penetration',  # Move probe down to target depth
    'anchor_expansion',  # Expand anchor horizontally
    'tip_extension',     # Extend the tip downward
    'body_follows',      # Move body down to follow tip
    'complete'           # Test complete
]

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
    def __init__(self, probe_parts):
        super().__init__()
        self.probe_parts = probe_parts
        self.probe_impulse = 0.0

    def PostSolve(self, contact, impulse: b2ContactImpulse):
        total = sum(impulse.normalImpulses)
        bodies = (contact.fixtureA.body, contact.fixtureB.body)
        if any(part in bodies for part in self.probe_parts):
            self.probe_impulse += total

# ================== World Setup ==================
def create_split_probe(world):
    # Create left shaft
    left_shaft = world.CreateDynamicBody(
        position=(-SHAFT_W/4, PROBE_POS_Y),
        fixedRotation=True,
        fixtures=b2FixtureDef(
            shape=b2PolygonShape(box=(SHAFT_W/4, SHAFT_H/2)),
            density=1.0, friction=0.2
        )
    )
    
    # Create right shaft
    right_shaft = world.CreateDynamicBody(
        position=(SHAFT_W/4, PROBE_POS_Y),
        fixedRotation=True,
        fixtures=b2FixtureDef(
            shape=b2PolygonShape(box=(SHAFT_W/4, SHAFT_H/2)),
            density=1.0, friction=0.2
        )
    )
    
    # Create tip
    tip = world.CreateDynamicBody(
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
    
    # Disable gravity for all probe parts
    left_shaft.gravityScale = 0.0
    right_shaft.gravityScale = 0.0
    tip.gravityScale = 0.0
    
    return left_shaft, right_shaft, tip

def create_vertical_joint(world, body):
    # Create an invisible static ground at origin
    ground = world.CreateStaticBody(position=(0,0))

    pj = b2PrismaticJointDef()
    # Anchor at the body's start; axis = vertical
    pj.Initialize(ground, body, anchor=body.position, axis=(0, 1))

    pj.enableLimit = True
    pj.lowerTranslation = -CHAMBER_HEIGHT  # can go down by chamber height
    pj.upperTranslation = 0.0             # cannot go up past start

    pj.enableMotor = True
    pj.motorSpeed = 0.0
    pj.maxMotorForce = 500.0

    return world.CreateJoint(pj)

def create_anchor_joint(world, left, right):
    # Create a prismatic joint between left and right shafts
    # This allows them to move horizontally relative to each other
    pj = b2PrismaticJointDef()
    pj.Initialize(left, right, anchor=(0, left.position.y), axis=(1, 0))
    
    pj.enableLimit = True
    pj.lowerTranslation = 0.0  # Cannot get closer
    pj.upperTranslation = SHAFT_W * EXPANSION_RATIO  # Max expansion distance
    
    pj.enableMotor = True
    pj.motorSpeed = 0.0  # Start with no movement
    pj.maxMotorForce = 500.0
    
    return world.CreateJoint(pj)

# ================== Anchor Expansion Logic ==================
def update_anchor_expansion(anchor_joint, left, right, speed=0.02, max_dx=0.2):
    anchor_joint.motorSpeed = speed
    dx = abs(right.position.x - left.position.x)
    if dx >= max_dx:
        anchor_joint.motorSpeed = 0.0
        return True
    return False

# ================ Rendering =================
def draw_probe_parts(screen, left, right, tip):
    # Draw left shaft
    for fixture in left.fixtures:
        shape = fixture.shape
        if isinstance(shape, b2PolygonShape):
            vertices = [left.transform * v for v in shape.vertices]
            pts = [world_to_screen_pts(v) for v in vertices]
            pygame.draw.polygon(screen, (100, 100, 255), pts)
    
    # Draw right shaft
    for fixture in right.fixtures:
        shape = fixture.shape
        if isinstance(shape, b2PolygonShape):
            vertices = [right.transform * v for v in shape.vertices]
            pts = [world_to_screen_pts(v) for v in vertices]
            pygame.draw.polygon(screen, (100, 100, 255), pts)
    
    # Draw tip
    for fixture in tip.fixtures:
        shape = fixture.shape
        if isinstance(shape, b2PolygonShape):
            vertices = [tip.transform * v for v in shape.vertices]
            pts = [world_to_screen_pts(v) for v in vertices]
            pygame.draw.polygon(screen, (255, 100, 100), pts)

def draw_grains(screen, grains):
    for grain in grains:
        x, y = world_to_screen(grain.position.x, grain.position.y)
        pygame.draw.circle(screen, (150, 150, 150), (x, y), int(GRAIN_RADIUS * SCALE))

# ================== Video Recorder ==================
class VideoRecorder:
    def __init__(self, fps=60, output_path="cpt_simulation.mp4"):
        self.fps = fps
        self.output_path = output_path
        self.recording = False
        self.writer = None
        self.frame_count = 0
        print(f"Video recorder initialized. Output will be saved to {self.output_path}")
        
    def start_recording(self, width, height):
        if self.recording:
            return
            
        # Make sure output directory exists
        os.makedirs(os.path.dirname(self.output_path) if os.path.dirname(self.output_path) else '.', exist_ok=True)
        
        # Initialize the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))
        self.recording = True
        self.frame_count = 0
        print("Recording started")
        
    def add_frame(self, surface):
        if not self.recording:
            return
            
        # Convert Pygame surface to a numpy array for OpenCV
        frame = pygame.surfarray.array3d(surface)
        # Transpose to get the correct orientation
        frame = frame.transpose([1, 0, 2])
        # Convert from RGB to BGR (OpenCV uses BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        self.writer.write(frame)
        self.frame_count += 1
        
        if self.frame_count % self.fps == 0:  # Log every second
            print(f"Recorded {self.frame_count} frames ({self.frame_count/self.fps:.1f} seconds)")
    
    def stop_recording(self):
        if not self.recording:
            return
            
        self.recording = False
        if self.writer:
            self.writer.release()
            self.writer = None
        print(f"Recording stopped. {self.frame_count} frames saved to {self.output_path}")

# ================ CPT Test ==================
def run_cpt_test():
    pygame.init()
    screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
    pygame.display.set_caption("CPT Test")
    clock = pygame.time.Clock()
    
    # Initialize video recorder
    recorder = VideoRecorder(fps=FPS, output_path="recordings/cpt_simulation.mp4")

    # Create world with increased downward gravity
    world = b2World(gravity=(0.0, -5.0))  # Increased from -2.0 to -5.0
    chamber = create_chamber(world)
    grains = create_grains(world)
    
    # Create the split probe (left shaft, right shaft, and tip)
    left_shaft, right_shaft, tip = create_split_probe(world)
    
    # Create vertical joints for both shafts
    joint_left = create_vertical_joint(world, left_shaft)
    joint_right = create_vertical_joint(world, right_shaft)
    
    # Create anchor joint between left and right shafts
    anchor_joint = create_anchor_joint(world, left_shaft, right_shaft)
    
    # Update contact detector for all probe parts
    detector = ContactDetector([left_shaft, right_shaft, tip])
    world.contactListener = detector

    pygame.font.init()
    font = pygame.font.Font(None, 24)

    # --- Driving parameters ---
    DRIVE_SPEED = 0.0  # Start with zero speed
    MAX_DRIVE_SPEED = TIP_EXTENSION_SPEED
    SPEED_INCREMENT = 0.001  # Gradual speed increase
    initial_y = left_shaft.position.y
    start_time = time.time()
    
    # Settling phase parameters
    SETTLING_TIME = 3.0  # seconds to allow grains to settle
    settling_phase = True
    settling_start_time = time.time()
    
    # Penetration and expansion phases
    penetration_phase = False
    expansion_phase = False
    target_depth = CHAMBER_HEIGHT / 2  # Target depth for penetration
    
    print("Starting grain settling phase...")
    print(f"Press 'R' to start/stop recording")

    running = True
    frame_count = 0
    
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                # Toggle recording
                if not recorder.recording:
                    recorder.start_recording(VIEWPORT_W, VIEWPORT_H)
                else:
                    recorder.stop_recording()
        
        current_time = time.time()
        
        # Handle settling phase
        if settling_phase:
            # Keep probe locked in place during settling
            joint_left.enableMotor = True
            joint_left.motorSpeed = 0.0
            joint_left.maxMotorForce = 1000.0  # Strong force to resist gravity
            
            joint_right.enableMotor = True
            joint_right.motorSpeed = 0.0
            joint_right.maxMotorForce = 1000.0
            
            # Check if settling time has elapsed
            if current_time - settling_start_time >= SETTLING_TIME:
                settling_phase = False
                penetration_phase = True
                print("Settling phase complete. Starting cone penetration...")
                print(f"Initial position: {left_shaft.position.y}")
        
        # Handle penetration phase
        elif penetration_phase:
            # Gradually increase speed
            if DRIVE_SPEED < MAX_DRIVE_SPEED:
                DRIVE_SPEED += SPEED_INCREMENT
                if DRIVE_SPEED > MAX_DRIVE_SPEED:
                    DRIVE_SPEED = MAX_DRIVE_SPEED
            
            # Drive both shafts downward in sync
            joint_left.enableMotor = True
            joint_left.motorSpeed = -DRIVE_SPEED   # negative → down
            joint_left.maxMotorForce = 500.0       # stall force
            
            joint_right.enableMotor = True
            joint_right.motorSpeed = -DRIVE_SPEED  # negative → down
            joint_right.maxMotorForce = 500.0      # stall force
            
            # Check if target depth is reached
            penetration_depth = initial_y - left_shaft.position.y
            if penetration_depth >= target_depth:
                penetration_phase = False
                expansion_phase = True
                # Stop vertical movement
                joint_left.motorSpeed = 0.0
                joint_right.motorSpeed = 0.0
                print("Target depth reached. Starting anchor expansion...")
        
        # Handle expansion phase
        elif expansion_phase:
            # Update anchor expansion
            expansion_complete = update_anchor_expansion(
                anchor_joint, 
                left_shaft, 
                right_shaft, 
                speed=0.02, 
                max_dx=0.2
            )
            
            if expansion_complete:
                expansion_phase = False
                print("Anchor expansion complete. Test finished.")
        
        # Log progress
        frame_count += 1
        if frame_count % 60 == 0:  # Once per second at 60 FPS
            penetration_depth = initial_y - left_shaft.position.y
            if settling_phase:
                phase_text = "SETTLING"
            elif penetration_phase:
                phase_text = "PENETRATION"
            elif expansion_phase:
                phase_text = "EXPANSION"
            else:
                phase_text = "COMPLETE"
                
            print(f"Phase: {phase_text}, Depth: {penetration_depth:.3f}m, Velocity: {DRIVE_SPEED} m/s")
            print(f"Position: {left_shaft.position.y:.3f}")
            print(f"Contact forces: {detector.probe_impulse:.2f} N·s")
            if expansion_phase or not (settling_phase or penetration_phase):
                dx = abs(right_shaft.position.x - left_shaft.position.x)
                print(f"Expansion distance: {dx:.3f}m")

        # Step physics simulation
        world.Step(TIME_STEP, VEL_ITERS, POS_ITERS)
        
        # Reset impulse measurements after physics step
        detector.probe_impulse = 0.0
        
        # Display current state
        screen.fill((255, 255, 255))
        
        # Draw chamber walls
        for body in chamber:
            for fixture in body.fixtures:
                shape = fixture.shape
                if isinstance(shape, b2PolygonShape):
                    pts = [world_to_screen_pts(body.transform * v) for v in shape.vertices]
                    pygame.draw.polygon(screen, (50, 50, 50), pts)
        
        # Draw grains
        draw_grains(screen, grains)
        
        # Draw probe parts
        draw_probe_parts(screen, left_shaft, right_shaft, tip)

        # Display debugging information
        penetration_depth = initial_y - left_shaft.position.y
        if settling_phase:
            phase_text = "SETTLING"
        elif penetration_phase:
            phase_text = "PENETRATION"
        elif expansion_phase:
            phase_text = "EXPANSION"
        else:
            phase_text = "COMPLETE"
            
        text = font.render(f"Phase: {phase_text}", True, (0, 0, 0))
        screen.blit(text, (10, 10))
        depth_text = font.render(f"Penetration depth: {penetration_depth:.3f}m", True, (0, 0, 0))
        screen.blit(depth_text, (10, 40))
        force_text = font.render(f"Probe Force: {detector.probe_impulse:.1f} N·s", True, (0, 0, 0))
        screen.blit(force_text, (10, 70))
        velocity_text = font.render(f"Speed: {DRIVE_SPEED:.4f} m/s", True, (0, 0, 0))
        screen.blit(velocity_text, (10, 100))
        time_text = font.render(f"Elapsed time: {current_time - start_time:.1f}s", True, (0, 0, 0))
        screen.blit(time_text, (10, 130))
        
        # Show expansion distance during expansion phase
        if expansion_phase or not (settling_phase or penetration_phase):
            dx = abs(right_shaft.position.x - left_shaft.position.x)
            expansion_text = font.render(f"Expansion: {dx:.3f}m", True, (0, 0, 0))
            screen.blit(expansion_text, (10, 160))

        # Add recording indicator
        if recorder.recording:
            rec_text = font.render("● REC", True, (255, 0, 0))
            screen.blit(rec_text, (VIEWPORT_W - 80, 10))

        pygame.display.flip()
        
        # Add frame to recording if active
        if recorder.recording:
            recorder.add_frame(screen)
            
        clock.tick(FPS)

    # Clean up recording before exiting
    if recorder.recording:
        recorder.stop_recording()
        
    pygame.quit()

if __name__ == "__main__":
    run_cpt_test()
