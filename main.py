from itertools import product
import numpy as np
import pygame
from colour import Color


def generate_coordinates(width, height, centre, scale):
    x_values = np.arange(-(width / 2) + centre[0], (width / 2) + centre[0]) / scale
    y_values = np.arange(-(height / 2) + centre[1], (height / 2) + centre[1]) / scale
    return np.array(list(product(x_values, y_values)))


def mandelbrot(coordinates, max_iter):
    """
    Performs Interior Distance Estimation on points on the mandelbrot set
    :param coordinates: Coordinate set that z = z^2 + c is performed over
    :param max_iter: The maximum amount of iterations for checking whether z = z^2 + c diverges
    :return:
    """
    c = coordinates[:, 0] + coordinates[:, 1] * 1j
    z = np.full(shape=c.shape, fill_value=0 + 0j)
    z0 = np.copy(c)
    d_dc = np.full(shape=c.shape, fill_value=0 + 0j)
    d_dz = np.full(shape=c.shape, fill_value=0 + 0j)
    d2_dcdz = np.full(shape=c.shape, fill_value=0 + 0j)
    d2_dz2 = np.full(shape=c.shape, fill_value=0 + 0j)
    mask = np.full(shape=c.shape, fill_value=True)
    for _ in range(max_iter):
        z[mask] = (z[mask] ** 2) + c[mask]
        d_dc[mask] = 2 * z[mask] * d_dc[mask] + 1
        d_dz[mask] = 2 * z[mask]
        d2_dcdz[mask] = 4 * z[mask] * (2 * z[mask] * d_dc[mask] + 1) + 2
        d2_dz2[mask] = 4 * (z0[mask] ** 3) + z[mask]
        mask = np.abs(z) < 4
        mask[np.isnan(z)] = False

    top_half = 1 - np.abs(d_dz) ** 2
    fraction = d_dc / (1 - d_dz)
    bottom_half = 4 * np.abs(d2_dcdz + d2_dz2 * fraction)
    b = top_half / bottom_half

    return b


def perform_colouring(points, color_from, color_to):
    mask = points < 0
    hsl_from = Color(rgb=color_from)
    hsl_to = Color(rgb=color_to)
    gradient_sections = 300
    palette_colors = hsl_from.range_to(hsl_to, gradient_sections)
    palette_colors = np.array(list(map(lambda x: x.rgb, palette_colors)))

    k = 3000
    with np.errstate(divide='ignore'):
        color_table_index = k * np.log(0.98 + points)
        color_table_index[np.isnan(color_table_index)] = 0
        color_table_index[color_table_index < 0] = 0
        color_table_index[np.isinf(color_table_index)] = gradient_sections - 1
        color_table_index[color_table_index > gradient_sections] = gradient_sections - 1
        color_table_index = color_table_index.astype(int)
        color_table_index2 = color_table_index - 1

    color_table_index2[color_table_index2 >= gradient_sections] = gradient_sections - 1
    current_color = palette_colors[color_table_index]
    current_color2 = palette_colors[color_table_index2]
    new_color = np.array(current_color2 - current_color)
    color_percentage = points.reshape(-1, 1)
    new_color *= np.repeat(color_percentage, 3, axis=1)
    new_color += current_color
    new_color_list = 255 * new_color
    new_color_list[mask] = [0, 0, 0]

    return new_color_list


WIDTH, HEIGHT = (1920, 1080)
iterations = 1000
colour_2 = (250 / 255, 179 / 255, 57 / 255)
colour_1 = (224 / 255, 34 / 255, 4 / 255)
pixel_set = generate_coordinates(WIDTH, HEIGHT, [-2, 1], 500)
threads = []

output = mandelbrot(pixel_set, iterations)
bitmap = perform_colouring(output, colour_1, colour_2).reshape((WIDTH, HEIGHT, 3))
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)

pygame.surfarray.blit_array(screen, bitmap)
running = True
while running:
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            running = False

