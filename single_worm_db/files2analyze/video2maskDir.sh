for NAME in `ls videos*.txt`
do
	sed -e s/thecus/MaskedVideos/g $NAME > ${NAME/videos/masks}
done
