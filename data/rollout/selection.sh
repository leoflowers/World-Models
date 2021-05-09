ls | sort -R | tail -250 | while read file; do
	mv $file ../yo
done
