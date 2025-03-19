#!/bin/bash

for f in m1*; do echo $f; s=$( cat $f/halo/rockstar_dm/snapshot_indices.txt );  if (( 10#$s<500  )); then echo $s; cd /projects/b1026/isultan/fire3/$f/halo/rockstar_dm/; pwd; cd /projects/b1026/isultan/fire3; fi; echo; done