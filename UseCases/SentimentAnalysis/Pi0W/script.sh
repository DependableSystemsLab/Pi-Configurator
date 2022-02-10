
file="001.csv"

for type in "bag" "TFIDF"
do
	for arg1 in {1..3..1}
	do
		for arg2 in {1..3..1}
		do
			echo "starting ${file} ${type} ${arg1} ${arg2}"
			python3 b.textMidProcess.py $file $type $arg1 $arg2  
			wait
		done
	done	
done
echo "done file-${file}"


file="010.csv"

for type in "bag" "TFIDF"
do
	for arg1 in {1..3..1}
	do
		for arg2 in {1..3..1}
		do
			echo "starting ${file} ${type} ${arg1} ${arg2}"
			python3 b.textMidProcess.py $file $type $arg1 $arg2  
			wait
		done
	done	
done
echo "done file-${file}"

file="011.csv"

for type in "bag" "TFIDF"
do
	for arg1 in {1..3..1}
	do
		for arg2 in {1..3..1}
		do
			echo "starting ${file} ${type} ${arg1} ${arg2}"
			python3 b.textMidProcess.py $file $type $arg1 $arg2  
			wait
		done
	done	
done
echo "done file-${file}"

file="100.csv"

for type in "bag" "TFIDF"
do
	for arg1 in {1..3..1}
	do
		for arg2 in {1..3..1}
		do
			echo "starting ${file} ${type} ${arg1} ${arg2}"
			python3 b.textMidProcess.py $file $type $arg1 $arg2  
			wait
		done
	done	
done
echo "done file-${file}"

file="101.csv"

for type in "bag" "TFIDF"
do
	for arg1 in {1..3..1}
	do
		for arg2 in {1..3..1}
		do
			echo "starting ${file} ${type} ${arg1} ${arg2}"
			python3 b.textMidProcess.py $file $type $arg1 $arg2  
			wait
		done
	done	
done
echo "done file-${file}"

file="110.csv"

for type in "bag" "TFIDF"
do
	for arg1 in {1..3..1}
	do
		for arg2 in {1..3..1}
		do
			echo "starting ${file} ${type} ${arg1} ${arg2}"
			python3 b.textMidProcess.py $file $type $arg1 $arg2  
			wait
		done
	done	
done
echo "done file-${file}"

file="111.csv"

for type in "bag" "TFIDF"
do
	for arg1 in {1..3..1}
	do
		for arg2 in {1..3..1}
		do
			echo "starting ${file} ${type} ${arg1} ${arg2}"
			python3 b.textMidProcess.py $file $type $arg1 $arg2  
			wait
		done
	done	
done
echo "done file-${file}"


