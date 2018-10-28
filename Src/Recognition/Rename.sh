for dir in "1" "2" "3" 
  do
  num=0
  cd "$dir"
  for file in *.png
  do
    num="$(printf "%0${#max}d" "$((10#$num+1))")"
    mv "$file" "$dir$num.png"
  done
  cd ..
done