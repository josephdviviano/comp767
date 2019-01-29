#!/bin/bash

# p=0.7, grid=5x5
#./mdp.py --iteration policy
#./mdp.py --iteration modified -k 25
#./mdp.py --iteration value

# p=0.9, grid=5x5
#./mdp.py --iteration policy -p 0.9
#./mdp.py --iteration modified -k 25 -p 0.9
#./mdp.py --iteration value -p 0.9

# p=0.7, grid=50x50
./mdp.py --iteration policy --size 50
./mdp.py --iteration modified -k 25 --size 50
./mdp.py --iteration value --size 50

# p=0.9, grid=50x50
./mdp.py --iteration policy --size 50 -p 0.9
./mdp.py --iteration modified -k 25 --size 50 -p 0.9
./mdp.py --iteration value --size 50 -p 0.9

