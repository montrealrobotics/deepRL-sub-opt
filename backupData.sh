#!/bin/bash

rsync -av --progress -e ssh glen.berseth@mila:~/playground/cleanrl/runs/ runs/