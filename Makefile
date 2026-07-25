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
