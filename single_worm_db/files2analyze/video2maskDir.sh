for NAME in `ls videos*.txt`
do
	sed -e s/thecus/MaskedVideos/g $NAME | sed s/.avi/.hdf5/g > ${NAME/videos/masks}
done
