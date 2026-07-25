0:
	uv run ./src/A_stl.py
1:
	uv run ./src/B_cluster.py
2:
	uv run ./src/C1_forcasting_preprocessing.py
3:
	uv run ./src/C2_train.py

00:
	python3 ./src/A_stl.py
01:
	python3 ./src/B_cluster.py
02:
	python3 ./src/C1_forcasting_preprocessing.py
03:
	python3 ./src/C2_train.py
04:
	python3 ./src/C3.py


p1:
	python3 ./src/pipeline/A_one.py
p2:
	python3 ./src/pipeline/B_two.py
p3:
	python3 ./src/pipeline/C_three.py
p4:
	python3 ./src/pipeline/D_four.py
p5:
	python3 ./src/pipeline/E_five.py