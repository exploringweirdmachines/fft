import pygame
import numpy as np
from numpy.fft import fft
from math import pi, sin, cos
import random
import ast
# import re
# import json
# from svg.path import parse_path
# from xml.dom import minidom
# from pathlib import Path
# def extract_svg_path_data(svg_file):
#     with open(svg_file, 'r') as f:
#         content = f.read()
#         svg_dom = minidom.parseString(content)
#         path_strings = [path.getAttribute('d') for path in svg_dom.getElementsByTagName('path')]
#         return path_strings

# def convert_path_data_to_points(path_data, num_points):
#     path = parse_path(path_data)
#     points = []
#     for i in range(num_points):
#         pos = i / (num_points - 1)
#         point = path.point(pos)
#         points.append({'x': point.real, 'y': point.imag})
#     return points

# # Replace 'path_to_logo.svg' with the path to your SVG file
# svg_file = Path.cwd() / "chapter 4" / 'drawing.svg'
# path_strings = extract_svg_path_data(svg_file)

# all_points = []
# num_points_per_path = 100

# for path_data in path_strings:
#     points = convert_path_data_to_points(path_data, num_points_per_path)
#     all_points.extend(points)

# drawing = all_points


# import random

# def generate_random_polygon(num_vertices, width_range, height_range):
#     return [{'x': random.randint(width_range[0], width_range[1]),
#              'y': random.randint(height_range[0], height_range[1])}
#             for _ in range(num_vertices)]

# # Generate a random polygon with 10 vertices
# random_logo = generate_random_polygon(10, (100, 400), (100, 400))
# drawing = random_logo

WIDTH, HEIGHT = 800, 600
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Epicycles Visualization")
clock = pygame.time.Clock()


def dft(y):
    N = len(y)
    fourier = []
    for m in range(N):
        re = 0
        im = 0
        for n in range(N):
            phi = (2 * pi * m * n) / N
            re += y[n] * cos(phi)
            im -= y[n] * sin(phi)
        re = re / N
        im = im / N

        freq = m
        amp = np.sqrt(re * re + im * im)
        phase = np.arctan2(im, re)
        fourier.append({'re': re, 'im': im, 'freq': freq, 'amp': amp, 'phase': phase})

    return sorted(fourier, key=lambda x: -x['amp'])

# Create the drawing array, x, y, and path lists
#drawing = []  # Replace with your own drawing data

#drawing = [{'x': random.randint(0, WIDTH), 'y': random.randint(0, HEIGHT)} for _ in range(200)]

drawing = [{'x': random.randint(0, WIDTH), 'y': random.randint(0, HEIGHT)} for _ in range(200)]

# Add these lines to load points from the file
with open('points.txt', 'r') as file:
    lines = file.readlines()

drawing = []
for line in lines:
    x, y = ast.literal_eval(line.strip())
    drawing.append({'x': x, 'y': y})

x = []
y = []
path = []
skip = 8

for i in range(0, len(drawing), skip):
    x.append(drawing[i]['x'])
    y.append(drawing[i]['y'])

fourierX = dft(x)
fourierY = dft(y)

fourierX.sort(key=lambda a: -a['amp'])
fourierY.sort(key=lambda a: -a['amp'])

time = 0

SCALE = 1

def epicycles(x, y, rotation, fourier, time):
    for i in range(len(fourier)):
        if fourier[i]['freq'] == 0:
            continue
        prevx = x
        prevy = y
        freq = fourier[i]['freq']
        radius = fourier[i]['amp'] * SCALE
        phase = fourier[i]['phase']
        x += radius * np.cos(freq * time + phase + rotation)
        y += radius * np.sin(freq * time + phase + rotation)

        pygame.draw.circle(screen, (255, 255, 255, 100), (int(prevx), int(prevy)), int(radius), 1)
        pygame.draw.line(screen, (255, 255, 255), (int(prevx), int(prevy)), (int(x), int(y)), 1)

    return x, y


def main():
    global time
    path = []
    running = True
    while running:
        screen.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        vx, vy = epicycles(WIDTH // 2 + 10, 10, 0, fourierX, time)
        vyx, vyy = epicycles(10, HEIGHT // 2 + 10, np.pi / 2, fourierY, time)

        v = (vx, vyy)
        path.insert(0, v)
        pygame.draw.line(screen, (255, 255, 255), (int(vx), int(vy)), v, 1)
        pygame.draw.line(screen, (255, 255, 255), (int(vyx), int(vyy)), v, 1)

        if len(path) >= 2:
            pygame.draw.lines(screen, (255, 255, 255), False, path)

        dt = 2 * np.pi / len(fourierY)
        time += 0.001  # Update the time variable here

        if time > 2 * np.pi:
            time = 0
            path = []

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()




# The Fast Fourier Transform (FFT) is already an efficient algorithm for computing the Discrete Fourier Transform (DFT) and its inverse. 
# The complexity of the FFT algorithm is O(n log n), which is a significant improvement over the naive DFT algorithm, which has a complexity of O(n^2).

# However, there might be further optimizations or specific use cases where an existing FFT implementation could be improved. Some ideas for improving the FFT include:

# Leveraging specialized hardware: GPUs, TPUs, or FPGAs can be used to parallelize and speed up the computation of the FFT. Some libraries like cuFFT (for NVIDIA GPUs) or clFFT (for OpenCL devices) 
# are already designed to take advantage of such hardware.

# Using machine learning: It's possible that machine learning methods, like deep learning, could be used to approximate the FFT for specific use cases. These methods might not provide exact results but could be faster 
# for certain applications where an approximate result is sufficient.

# Exploiting sparsity: If the input data has some special structure, like being sparse, it might be possible to develop more efficient algorithms tailored to that specific structure.

# Optimizing for specific platforms: Some FFT libraries are optimized for specific platforms or processors. If you know the target platform for your application, you could optimize the FFT implementation to take advantage 
# of the specific architecture and its features.

# Regarding PyTorch and tensors, PyTorch already has an FFT implementation built on top of cuFFT (for GPU) and MKL-DNN (for CPU). You can use torch.fft module in PyTorch to compute the FFT on tensors. 
# This module can take advantage of GPU acceleration, which could result in a performance improvement for large data sets.

# It's important to note that improving FFT algorithms is an active area of research, and new optimizations and methods may be discovered in the future. However, the current FFT algorithms and their 
# implementations in various libraries are already quite efficient and optimized for most use cases.