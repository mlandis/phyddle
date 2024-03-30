PDF_FN=$1
PNG_FN="$(basename ${PDF_FN} .pdf).png"
convert -density 300 ${PDF_FN} -quality 95 ${PNG_FN}
