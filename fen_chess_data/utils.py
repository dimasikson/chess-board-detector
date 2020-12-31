import cv2
import os

import pandas as pd
import numpy as np
from scipy import ndimage
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import chess
import chess.svg

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

def distance(p1, p2):
    return math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )

def clipped_normal(den=20):
    nml = np.random.normal(0, 1/den)
    return np.clip(nml, 0, float('inf'))

def callback_test_error(labels, preds):

    den = num = 0
    preds = np.array(preds)

    for p, l in zip(preds, labels):

        den += 1

        if np.argmax(p) == np.argmax(l):
            num += 1

    return num / den

def generate_img(fen, large_dim=256, i_board=1, i_piece=1):

    folder = 'C:/Users/dimaz/Documents/chessComputerVisionProject/fen_chess_data/chess-generator/ChessGenerator'

    # generate img
    board = cv2.imread(f"{folder}/boards/{i_board}.png")

    # draw fen
    fen = fen_to_labels(fen)

    for i, piece in enumerate(fen):
        col = i % 8
        row = (i - col) // 8

        # skip if no piece
        if piece == '0': continue

        wb = 'w' if piece.isupper() else 'b'
        piece = piece.lower()

        img = cv2.imread(f"{folder}/pieces/{i_piece}/{piece}_{wb}.png")
        board[50*row:50*(row+1), 50*col:50*(col+1), :] = paint_square(board[50*row:50*(row+1), 50*col:50*(col+1), :], img)
 
    board = cv2.resize( board, (large_dim, large_dim), interpolation = cv2.INTER_AREA )

    return board

def paint_square(bg, fg):

    if bg.shape != fg.shape:
        return None

    ts_color = fg[0,0,:]
    w, h, c = bg.shape
    out = np.zeros((w,h,c))

    for i in range(w):
        for j in range(h):

            if np.array_equal( fg[i, j, :], ts_color ):
                out[i, j, :] = bg[i, j, :]

            else:
                out[i, j, :] = fg[i, j, :]

    return np.int16( out )

def find_outer_corners(img, pts):

    rows, cols, _ = img.shape

    bl_dst = br_dst = tl_dst = tr_dst = float('inf')

    for p in pts:

        p = p[0]

        if distance(p, (cols*0, rows*1)) < bl_dst:
            bl_dst = distance(p, (cols*0, rows*1))
            bl = p

        if distance(p, (cols*1, rows*1)) < br_dst:
            br_dst = distance(p, (cols*1, rows*1))
            br = p

        if distance(p, (cols*0, rows*0)) < tl_dst:
            tl_dst = distance(p, (cols*0, rows*0))
            tl = p

        if distance(p, (cols*1, rows*0)) < tr_dst:
            tr_dst = distance(p, (cols*1, rows*0))
            tr = p

    pts1 = np.float32(
        [bl,  # btm left
        br,  # btm right
        tl,  # top left
        tr]  # top right
    )

    return pts1

def find_max_contour_area(contours):

    max_area = 0 - float('inf')
    max_c = None

    for c in contours:
        area = cv2.contourArea(c)

        if area > max_area:
            max_area = area
            max_c = c

    return [max_c]

def do_perspective_transform(img, pts, pts_type=1):

    rows, cols = img.shape[:2]

    bl = [cols*0, rows*1] 
    br = [cols*1, rows*1] 
    tl = [cols*0, rows*0] 
    tr = [cols*1, rows*0] 

    pts2 = np.float32([bl, br, tl, tr])

    if pts_type == 2:
        pts, pts2 = pts2, pts

    M = cv2.getPerspectiveTransform(pts,pts2)

    color = 255

    if len(img.shape) == 3:
        color = (255, 255, 255)

    img = cv2.warpPerspective(img, M, (cols, rows), borderValue=color)

    return img

def random_warp(img, thr=2, desaturation_weight=0.5):

    rows, cols = img.shape[:2]

    bl = [cols * ( 0 + clipped_normal() ), rows * ( 1 - clipped_normal() )]  # btm left
    br = [cols * ( 1 - clipped_normal() ), rows * ( 1 - clipped_normal() )]  # btm right
    tl = [cols * ( 0 + clipped_normal() ), rows * ( 0 + clipped_normal() )]  # top left
    tr = [cols * ( 1 - clipped_normal() ), rows * ( 0 + clipped_normal() )]  # top right

    pts = np.float32([bl, br, tl, tr])

    img = do_perspective_transform(img, pts, pts_type=2)

    for i in range(len(pts)):
        for j in range(len(pts[i])):
            pts[i][j] += np.random.randint(low=-thr, high=thr)

    img = do_perspective_transform(img, pts, pts_type=1)

    # desaturation
    shape = (rows,cols,3)[:len(img.shape)]
    overlay = np.zeros(shape) + 127

    img = cv2.addWeighted(
        np.int32( img ), 1-desaturation_weight,
        np.int32( overlay ), desaturation_weight,
        0
    )

    return img

def fen_to_labels(fen):

    fen = ''.join(['0' * int(c) if c.isdigit() else c for c in fen])
    fen = fen.replace('/', '')

    labels = [c for c in fen]

    return labels

def preds_to_fen(preds):

    pieces = {i:c for i, c in enumerate('0PRNBQKprnbqk')}

    fen = ''

    for i, p in enumerate(preds):
        num = np.argmax(p)
        fen += pieces[num]

        i += 1
        if i % 8 == 0 and i < 64:
            fen += '/'

    for i in range(8, 0, -1):
        fen = fen.replace('0' * i, str(i))

    return fen

def split_chessboard(img):

    w = img.shape[0]
    sq = w // 8

    imgs = []

    for i in range(0,w,sq):
        for j in range(0,w,sq):
            imgs.append( img[ i : i + sq, j : j + sq ] )

    return imgs
