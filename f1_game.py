import math
import numpy as np
import pygame
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Car:
    x: float
    y: float
    angle: float
    speed: float
    max_speed: float = 8.0
    acceleration: float = 0.3
    brake_power: float = 0.5
    turn_speed: float = 0.08
    friction: float = 0.02


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float = 15.0


@dataclass
class Checkpoint:
    x: float 
    y: float
    radius: float = 40.0
    passed: bool = False


class F1Game:
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (100, 100, 100)
    RED = (255, 50, 50)
    GREEN = (50, 255, 50)
    BLUE = (50, 100, 255)
    YELLOW = (255, 255, 50)
    ORANGE = (255, 165, 0)
    
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.render_enabled = False
        
        self.track_center_x = width // 2
        self.track_center_y = height // 2
        self.track_width = 80
        self.outer_radius_x = 320
        self.outer_radius_y = 220
        self.inner_radius_x = self.outer_radius_x - self.track_width
        self.inner_radius_y = self.outer_radius_y - self.track_width
        
        self.car: Optional[Car] = None
        self.obstacles: List[Obstacle] = []
        self.checkpoints: List[Checkpoint] = []
        self.current_checkpoint: int = 0
        self.lap_count: int = 0
        self.steps: int = 0
        self.max_steps: int = 2000
        
        self.num_sensors = 5
        self.sensor_angles = [-60, -30, 0, 30, 60]
        self.sensor_length = 150
        
        self._setup_track()
        
    def _setup_track(self):
        num_checkpoints = 8
        self.checkpoints = []
        for i in range(num_checkpoints):
            angle = 2 * math.pi * i / num_checkpoints
            mid_radius_x = (self.outer_radius_x + self.inner_radius_x) / 2
            mid_radius_y = (self.outer_radius_y + self.inner_radius_y) / 2
            x = self.track_center_x + mid_radius_x * math.cos(angle)
            y = self.track_center_y + mid_radius_y * math.sin(angle)
            self.checkpoints.append(Checkpoint(x=x, y=y))
        self._generate_obstacles()
    
    def _generate_obstacles(self):
        self.obstacles = []
        num_obstacles = 6
        for i in range(num_obstacles):
            angle = 2 * math.pi * (i + 0.5) / num_obstacles
            offset = np.random.uniform(-20, 20)
            mid_radius_x = (self.outer_radius_x + self.inner_radius_x) / 2 + offset
            mid_radius_y = (self.outer_radius_y + self.inner_radius_y) / 2 + offset
            x = self.track_center_x + mid_radius_x * math.cos(angle)
            y = self.track_center_y + mid_radius_y * math.sin(angle)
            self.obstacles.append(Obstacle(x=x, y=y, radius=12))
    
    def reset(self) -> np.ndarray:
        start_x = self.track_center_x + (self.outer_radius_x + self.inner_radius_x) / 2
        start_y = self.track_center_y
        
        self.car = Car(x=start_x, y=start_y, angle=-math.pi / 2, speed=0.0)
        
        for cp in self.checkpoints:
            cp.passed = False
        self.current_checkpoint = 0
        self.lap_count = 0
        self.steps = 0
        self._generate_obstacles()
        
        return self.get_state()
    
    def step(self, action: int) -> Tuple[float, bool, dict]:
        self.steps += 1
        old_checkpoint = self.current_checkpoint
        
        if action == 0:
            self.car.speed = min(self.car.speed + self.car.acceleration, self.car.max_speed)
        elif action == 1:
            self.car.angle -= self.car.turn_speed
            self.car.speed = min(self.car.speed + self.car.acceleration * 0.5, self.car.max_speed)
        elif action == 2:
            self.car.angle += self.car.turn_speed
            self.car.speed = min(self.car.speed + self.car.acceleration * 0.5, self.car.max_speed)
        elif action == 3:
            self.car.speed = max(self.car.speed - self.car.brake_power, 0)
        
        self.car.speed = max(self.car.speed - self.car.friction, 0)
        self.car.x += self.car.speed * math.cos(self.car.angle)
        self.car.y += self.car.speed * math.sin(self.car.angle)
        
        reward = -0.1
        done = False
        
        if self._check_wall_collision():
            reward = -100
            done = True
            return reward, done, {'score': self.lap_count, 'checkpoints': self.current_checkpoint}
        
        if self._check_obstacle_collision():
            reward = -100
            done = True
            return reward, done, {'score': self.lap_count, 'checkpoints': self.current_checkpoint}
        
        if self._check_checkpoint():
            reward += 50
            if self.current_checkpoint == 0 and old_checkpoint > 0:
                self.lap_count += 1
                reward += 200
        
        next_cp = self.checkpoints[self.current_checkpoint]
        dist_to_cp = math.sqrt((self.car.x - next_cp.x)**2 + (self.car.y - next_cp.y)**2)
        if dist_to_cp < 100:
            reward += 1
        
        if self.steps >= self.max_steps:
            done = True
        
        return reward, done, {'score': self.lap_count, 'checkpoints': self.current_checkpoint, 'steps': self.steps}
    
    def _check_wall_collision(self) -> bool:
        dx = self.car.x - self.track_center_x
        dy = self.car.y - self.track_center_y
        outer_normalized = (dx / self.outer_radius_x)**2 + (dy / self.outer_radius_y)**2
        inner_normalized = (dx / self.inner_radius_x)**2 + (dy / self.inner_radius_y)**2
        return outer_normalized > 1.0 or inner_normalized < 1.0
    
    def _check_obstacle_collision(self) -> bool:
        car_radius = 10
        for obs in self.obstacles:
            dist = math.sqrt((self.car.x - obs.x)**2 + (self.car.y - obs.y)**2)
            if dist < (car_radius + obs.radius):
                return True
        return False
    
    def _check_checkpoint(self) -> bool:
        cp = self.checkpoints[self.current_checkpoint]
        dist = math.sqrt((self.car.x - cp.x)**2 + (self.car.y - cp.y)**2)
        if dist < cp.radius:
            cp.passed = True
            self.current_checkpoint = (self.current_checkpoint + 1) % len(self.checkpoints)
            return True
        return False
    
    def get_state(self) -> np.ndarray:
        state = np.zeros(15, dtype=np.float32)
        
        for i, angle_offset in enumerate(self.sensor_angles):
            sensor_angle = self.car.angle + math.radians(angle_offset)
            dist = self._cast_ray_to_wall(sensor_angle)
            state[i] = dist / self.sensor_length
        
        vx = self.car.speed * math.cos(self.car.angle)
        vy = self.car.speed * math.sin(self.car.angle)
        state[5] = vx / self.car.max_speed
        state[6] = vy / self.car.max_speed
        state[7] = self.car.angle / math.pi
        
        next_cp = self.checkpoints[self.current_checkpoint]
        dx = next_cp.x - self.car.x
        dy = next_cp.y - self.car.y
        dist = max(math.sqrt(dx**2 + dy**2), 1)
        state[8] = dx / dist
        state[9] = dy / dist
        
        for i, angle_offset in enumerate(self.sensor_angles):
            sensor_angle = self.car.angle + math.radians(angle_offset)
            dist = self._cast_ray_to_obstacle(sensor_angle)
            state[10 + i] = dist / self.sensor_length
        
        return state
    
    def _cast_ray_to_wall(self, angle: float) -> float:
        for d in range(1, self.sensor_length + 1, 5):
            x = self.car.x + d * math.cos(angle)
            y = self.car.y + d * math.sin(angle)
            dx = x - self.track_center_x
            dy = y - self.track_center_y
            outer_normalized = (dx / self.outer_radius_x)**2 + (dy / self.outer_radius_y)**2
            inner_normalized = (dx / self.inner_radius_x)**2 + (dy / self.inner_radius_y)**2
            if outer_normalized > 1.0 or inner_normalized < 1.0:
                return float(d)
        return float(self.sensor_length)
    
    def _cast_ray_to_obstacle(self, angle: float) -> float:
        min_dist = float(self.sensor_length)
        for obs in self.obstacles:
            dx = obs.x - self.car.x
            dy = obs.y - self.car.y
            ray_dx = math.cos(angle)
            ray_dy = math.sin(angle)
            proj = dx * ray_dx + dy * ray_dy
            if proj > 0 and proj < self.sensor_length:
                perp_dist = abs(dx * ray_dy - dy * ray_dx)
                if perp_dist < obs.radius:
                    min_dist = min(min_dist, proj)
        return min_dist
    
    def enable_render(self):
        if not self.render_enabled:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("F1 Racing RL")
            self.clock = pygame.time.Clock()
            self.render_enabled = True
    
    def disable_render(self):
        if self.render_enabled:
            pygame.quit()
            self.render_enabled = False
            self.screen = None
            self.clock = None
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        if not self.render_enabled:
            self.enable_render()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.disable_render()
                return None
        
        self.screen.fill(self.GRAY)
        
        pygame.draw.ellipse(self.screen, self.BLACK,
            (self.track_center_x - self.outer_radius_x,
             self.track_center_y - self.outer_radius_y,
             self.outer_radius_x * 2, self.outer_radius_y * 2), 5)
        
        pygame.draw.ellipse(self.screen, self.BLACK,
            (self.track_center_x - self.inner_radius_x,
             self.track_center_y - self.inner_radius_y,
             self.inner_radius_x * 2, self.inner_radius_y * 2), 5)
        
        for i, cp in enumerate(self.checkpoints):
            color = self.GREEN if cp.passed else self.YELLOW
            if i == self.current_checkpoint:
                color = self.ORANGE
            pygame.draw.circle(self.screen, color, (int(cp.x), int(cp.y)), int(cp.radius), 2)
        
        for obs in self.obstacles:
            pygame.draw.circle(self.screen, self.RED, (int(obs.x), int(obs.y)), int(obs.radius))
        
        if self.car:
            car_length = 20
            car_width = 10
            cos_a = math.cos(self.car.angle)
            sin_a = math.sin(self.car.angle)
            
            front = (self.car.x + car_length * cos_a, self.car.y + car_length * sin_a)
            back_left = (self.car.x - car_width * sin_a - car_length/2 * cos_a,
                        self.car.y + car_width * cos_a - car_length/2 * sin_a)
            back_right = (self.car.x + car_width * sin_a - car_length/2 * cos_a,
                         self.car.y - car_width * cos_a - car_length/2 * sin_a)
            
            pygame.draw.polygon(self.screen, self.BLUE, [front, back_left, back_right])
            
            for angle_offset in self.sensor_angles:
                sensor_angle = self.car.angle + math.radians(angle_offset)
                end_x = self.car.x + self.sensor_length * 0.5 * math.cos(sensor_angle)
                end_y = self.car.y + self.sensor_length * 0.5 * math.sin(sensor_angle)
                pygame.draw.line(self.screen, (100, 100, 200), 
                               (int(self.car.x), int(self.car.y)),
                               (int(end_x), int(end_y)), 1)
        
        font = pygame.font.Font(None, 36)
        lap_text = font.render(f"Laps: {self.lap_count}", True, self.WHITE)
        cp_text = font.render(f"Checkpoint: {self.current_checkpoint}/{len(self.checkpoints)}", True, self.WHITE)
        speed_text = font.render(f"Speed: {self.car.speed:.1f}", True, self.WHITE)
        self.screen.blit(lap_text, (10, 10))
        self.screen.blit(cp_text, (10, 45))
        self.screen.blit(speed_text, (10, 80))
        
        pygame.display.flip()
        
        if mode == "rgb_array":
            return pygame.surfarray.array3d(self.screen)
        return None
    
    def render_with_fps(self, fps: int = 60):
        self.render()
        if self.clock:
            self.clock.tick(fps)
