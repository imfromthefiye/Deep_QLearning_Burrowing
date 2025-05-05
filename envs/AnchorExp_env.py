import pygame
from Box2D import (
    b2World, b2PolygonShape, b2FixtureDef, b2CircleShape,
    b2PrismaticJointDef, b2ContactListener, b2ContactImpulse, b2MouseJointDef
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
PROBE_POS_Y = soil_height   # Only small clearance above soil
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
    
    # Create tip - position it exactly at the bottom of shafts
    tip_y_pos = PROBE_POS_Y - SHAFT_H/2 - TIP_HEIGHT/2
    tip = world.CreateDynamicBody(
        position=(0.0, tip_y_pos),
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
    
    # Create weld joints to connect tip to both shaft halves
    world.CreateWeldJoint(
        bodyA=left_shaft,
        bodyB=tip,
        anchor=(left_shaft.position.x, tip_y_pos + TIP_HEIGHT/2),
        referenceAngle=0.0
    )
    
    world.CreateWeldJoint(
        bodyA=right_shaft,
        bodyB=tip,
        anchor=(right_shaft.position.x, tip_y_pos + TIP_HEIGHT/2),
        referenceAngle=0.0
    )
    
    return left_shaft, right_shaft, tip

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

def fix_probe_position(world, ground, left, right, tip):
    """Create temporary mouse joints to fix probe position"""
    # Create mouse joints to temporarily fix each part in place
    mj_left = world.CreateJoint(b2MouseJointDef(
        bodyA=ground,
        bodyB=left,
        target=left.position,
        maxForce=1000.0 * left.mass,
        frequencyHz=100.0,
        dampingRatio=1.0
    ))
    
    mj_right = world.CreateJoint(b2MouseJointDef(
        bodyA=ground,
        bodyB=right,
        target=right.position,
        maxForce=1000.0 * right.mass,
        frequencyHz=100.0,
        dampingRatio=1.0
    ))
    
    mj_tip = world.CreateJoint(b2MouseJointDef(
        bodyA=ground,
        bodyB=tip,
        target=tip.position,
        maxForce=1000.0 * tip.mass,
        frequencyHz=100.0,
        dampingRatio=1.0
    ))
    
    return [mj_left, mj_right, mj_tip]

def main():
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
    pygame.display.set_caption("Anchore Expansion Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 16)
    
    # Create world with gravity
    world = b2World(gravity=(0, -9.81))
    
    # Create explicit ground body
    ground = world.CreateStaticBody(position=(0, 0))
    
    # Create chamber walls
    walls = create_chamber(world)
    
    # Create probe parts using the create_split_probe function
    left_shaft, right_shaft, tip = create_split_probe(world)
    
    # Get probe positions and dimensions for collision avoidance
    probe_x_min = min(left_shaft.position.x - SHAFT_W/4, tip.position.x - TIP_BASE/2)
    probe_x_max = max(right_shaft.position.x + SHAFT_W/4, tip.position.x + TIP_BASE/2)
    probe_y_min = tip.position.y - TIP_HEIGHT/2
    probe_y_max = max(left_shaft.position.y + SHAFT_H/2, right_shaft.position.y + SHAFT_H/2)
    
    # Create grains avoiding the probe area
    grains = []
    grain_diameter = GRAIN_RADIUS * 2
    COLS = int(CHAMBER_WIDTH / grain_diameter) + 2
    soil_height = CHAMBER_HEIGHT / 3
    ROWS = int(soil_height / (grain_diameter * 0.84))
    x0 = -(COLS-1) * grain_diameter / 2
    y0 = GRAIN_RADIUS
    
    for i in range(ROWS):
        for j in range(COLS):
            x = x0 + j * grain_diameter + (0.5 * grain_diameter if i % 2 else 0)
            y = y0 + i * (grain_diameter * 0.84)
            
            # Add random jitter
            x += np.random.uniform(-0.002, 0.002)
            y += np.random.uniform(-0.002, 0.002)
            
            # Skip if grain would be outside chamber
            if abs(x) > (CHAMBER_WIDTH/2 + GRAIN_RADIUS/2):
                continue
                
            # Skip if grain would overlap with probe (add small buffer)
            buffer = GRAIN_RADIUS * 1.1
            if (x + buffer > probe_x_min and x - buffer < probe_x_max and 
                y + buffer > probe_y_min and y - buffer < probe_y_max):
                continue
                
            body = world.CreateDynamicBody(
                position=(x, y),
                fixtures=b2FixtureDef(
                    shape=b2CircleShape(radius=GRAIN_RADIUS),
                    density=2.5,
                    friction=0.6,
                    restitution=0.0
                ),
                linearDamping=0.5,
                angularDamping=0.6
            )
            body.bullet = True
            grains.append(body)
    
    # Create the anchor joint between shafts
    anchor_joint = create_anchor_joint(world, left_shaft, right_shaft)
    
    # Fix probe in place
    mouse_joints = fix_probe_position(world, ground, left_shaft, right_shaft, tip)
    
    # Expansion parameters
    EXPANSION_DISTANCE = 0.2  # Maximum expansion distance for each half (D)
    EXPANSION_SPEED = 0.05    # Speed of expansion (m/s)
    expansion_started = False
    expansion_complete = False
    
    # Start simulation loop with settling time
    running = True
    settling_time = 2.0  # seconds
    start_time = time.time()
    settling_end_time = start_time + settling_time
    
    while running:
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE and current_time >= settling_end_time and not expansion_started:
                    # Start expansion on SPACE key after settling
                    for joint in mouse_joints:
                        world.DestroyJoint(joint)
                    mouse_joints = []
                    expansion_started = True
                    # Set motor speed to start expansion
                    anchor_joint.motorSpeed = EXPANSION_SPEED
        
        # Auto-start expansion after settling
        if not expansion_started and current_time >= settling_end_time + 1.0:
            for joint in mouse_joints:
                world.DestroyJoint(joint)
            mouse_joints = []
            expansion_started = True
            # Set motor speed to start expansion
            anchor_joint.motorSpeed = EXPANSION_SPEED
        
        # Check for expansion completion
        if expansion_started and not expansion_complete:
            # Get current translation - use the correct attribute
            translation = anchor_joint.translation
            
            # Check if we've reached the expansion limit
            if translation >= EXPANSION_DISTANCE:
                anchor_joint.motorSpeed = 0.0  # Stop the motor
                expansion_complete = True
        
        # Step physics
        world.Step(TIME_STEP, VEL_ITERS, POS_ITERS)
        
        # Render everything
        screen.fill((255, 255, 255))
        
        # Draw chamber walls
        for wall in walls:
            for fixture in wall.fixtures:
                shape = fixture.shape
                vertices = [(wall.transform * v) for v in shape.vertices]
                screen_vertices = [world_to_screen(v.x, v.y) for v in vertices]
                pygame.draw.polygon(screen, (50, 50, 50), screen_vertices)
        
        # Draw grains
        draw_grains(screen, grains)
        
        # Draw probe parts
        draw_probe_parts(screen, left_shaft, right_shaft, tip)
        
        # Get current translation for display
        translation = anchor_joint.translation
        
        # Display status text
        if current_time < settling_end_time:
            status = f"SETTLING: {settling_time - (current_time - start_time):.1f}s left"
        elif not expansion_started:
            status = f"READY: Press SPACE to start expansion"
        elif not expansion_complete:
            status = f"EXPANDING: {translation:.3f}m / {EXPANSION_DISTANCE:.3f}m"
        else:
            status = f"COMPLETE: {translation:.3f}m expansion reached"
        
        info_text = f"Time: {elapsed_time:.2f}s    Status: {status}"
        text_surface = font.render(info_text, True, (0, 0, 0))
        screen.blit(text_surface, (10, 10))
        
        # Update display
        pygame.display.flip()
        
        # Cap framerate
        clock.tick(FPS)
    
    pygame.quit()

if __name__ == "__main__":
    main()




