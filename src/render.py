"""Pygame rendering for the double pendulum on a cart: cart, rods, bobs, track limits."""

from __future__ import annotations

import numpy as np
import pygame

SCALE = 150.0
WIDTH = 800
HEIGHT = 400
CART_Y = 300
CART_W = 60
CART_H = 36
BOB_R = 9

TRACK_COLOR = (60, 60, 70)
LIMIT_COLOR = (160, 40, 40)
CART_COLOR = (70, 130, 220)
ROD_COLOR = (200, 200, 210)
BOB_COLOR = (240, 180, 60)
BG_COLOR = (18, 18, 24)


class PendulumRenderer:
    """Draws the system state. ``human`` opens a window; ``rgb_array`` renders offscreen."""

    def __init__(self, render_mode: str, track_limit: float, l1: float, l2: float):
        if render_mode not in ("human", "rgb_array"):
            raise ValueError(f"unsupported render_mode: {render_mode}")
        self.render_mode = render_mode
        self.track_limit = track_limit
        self.l1 = l1
        self.l2 = l2
        self.screen = None
        self._surface = pygame.Surface((WIDTH, HEIGHT))
        if render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Double pendulum on a cart")
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))

    def _world_to_screen(self, x_m: float, y_m: float) -> tuple[int, int]:
        return int(WIDTH / 2 + x_m * SCALE), int(CART_Y - y_m * SCALE)

    def _draw(self, surface: pygame.Surface, state: np.ndarray) -> None:
        x, th1, th2 = state[0], state[1], state[2]
        surface.fill(BG_COLOR)
        y_track = self._world_to_screen(0.0, 0.0)[1]
        left = self._world_to_screen(-self.track_limit, 0.0)[0]
        right = self._world_to_screen(self.track_limit, 0.0)[0]
        pygame.draw.line(surface, TRACK_COLOR, (left, y_track), (right, y_track), 3)
        for x_lim in (left, right):
            pygame.draw.line(surface, LIMIT_COLOR, (x_lim, y_track - 30), (x_lim, y_track + 10), 3)
        cx, cy = self._world_to_screen(x, 0.0)
        cart = pygame.Rect(0, 0, CART_W, CART_H)
        cart.center = (cx, cy + CART_H // 2)
        pygame.draw.rect(surface, CART_COLOR, cart, border_radius=4)
        bob1 = self._world_to_screen(x + self.l1 * np.sin(th1), self.l1 * np.cos(th1))
        bob2 = self._world_to_screen(
            x + self.l1 * np.sin(th1) + self.l2 * np.sin(th2),
            self.l1 * np.cos(th1) + self.l2 * np.cos(th2),
        )
        pygame.draw.line(surface, ROD_COLOR, (cx, cy), bob1, 4)
        pygame.draw.line(surface, ROD_COLOR, bob1, bob2, 4)
        pygame.draw.circle(surface, BOB_COLOR, bob1, BOB_R)
        pygame.draw.circle(surface, BOB_COLOR, bob2, BOB_R)

    def render(self, state: np.ndarray):
        """Return an rgb array (``rgb_array``) or blit to the window (``human``)."""
        if self.render_mode == "rgb_array":
            self._draw(self._surface, np.asarray(state))
            return np.transpose(pygame.surfarray.array3d(self._surface), (1, 0, 2))
        self._draw(self.screen, np.asarray(state))
        pygame.display.flip()
        return None

    def close(self) -> None:
        if self.screen is not None:
            pygame.quit()
            self.screen = None
