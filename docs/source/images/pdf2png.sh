PDF_FN=$1
PNG_FN="$(basename ${PDF_FN} .pdf).png"
convert -density 150 ${PDF_FN} -quality 90 ${PNG_FN}
