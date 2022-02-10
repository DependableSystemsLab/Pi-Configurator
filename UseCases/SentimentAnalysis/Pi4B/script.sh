
file="001.csv"

for type in "bag" "TFIDF"
do
	for arg1 in {1..3..1}
	do
		for arg2 in {1..3..1}
		do
			for model in "logis" "knn"
			do
				for param in {1..20..2}
				do
					echo "starting ${file} ${type} ${arg1} ${arg2} ${model} ${param} "
					python3 c.textModeling.py $file $type $arg1 $arg2 $model $param 
					wait	
				done
			done
			
		done
	done	
done

for type in "bag" "TFIDF"
do
	for arg1 in {1..3..1}
	do
		for arg2 in {1..3..1}
		do
			for model in "sgd"
			do
				for param in $(seq 0.0001 0.1 1.00)
				do
					echo "starting ${file} ${type} ${arg1} ${arg2} ${model} ${param} "
					python3 c.textModeling.py $file $type $arg1 $arg2 $model $param 
					wait	
				done
			done
			
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
			for model in "logis" "knn"
			do
				for param in {1..20..2}
				do
					echo "starting ${file} ${type} ${arg1} ${arg2} ${model} ${param} "
					python3 c.textModeling.py $file $type $arg1 $arg2 $model $param 
					wait	
				done
			done
			
		done
	done	
done

for type in "bag" "TFIDF"
do
	for arg1 in {1..3..1}
	do
		for arg2 in {1..3..1}
		do
			for model in "sgd"
			do
				for param in $(seq 0.0001 0.1 1.00)
				do
					echo "starting ${file} ${type} ${arg1} ${arg2} ${model} ${param} "
					python3 c.textModeling.py $file $type $arg1 $arg2 $model $param 
					wait	
				done
			done
			
		done
	done	
done

echo "done file-${file}"
