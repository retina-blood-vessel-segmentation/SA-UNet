#!/bin/bash
#Script needed to make this folder, once checked out, ready for use on any of the cluster computers.
# It will create the symlinks needed for all the data, the models and all that. 
ln -s /home/shared/retina/models/saunet/trained models
ln -s /home/shared/retina/staging data
ln -s /home/shared/retina/output/saunet results