#!/bin/bash
#########################################
#############VARIABLES###################
#########################################
path_h5="/home/rspezialetti/Desktop/4flow_5000points_reconstruction.h5"
path_png="/home/rspezialetti/Desktop/figures/"
path_mitsuba="/home/rspezialetti/lib/mitsuba2/build/dist/"
name_png="4f"

python render_png.py --path_h5 $path_h5 --path_png $path_png --path_mitsuba $path_mitsuba --name_png $name_png --indices 1 10 22
