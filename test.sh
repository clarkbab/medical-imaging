join () {
  local IFS=","
  shift
  echo "$@"
}

list=('a' 'b' 'c')

test=$(join "${list[@]}")
echo $test

