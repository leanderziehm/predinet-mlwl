# Predinet-Github


## Setup:

```
python3 -m venv .venv
source ./.venv/bin/activate
make all
```

Makefile:
```
all:
	p1 p2 p3 p4 p5
p1:
	python3 ./src/pipeline/A_cell_forecastability_features.py
p2:
	python3 ./src/pipeline/B_cluster.py
p3:
	python3 ./src/pipeline/C_prepare_training_data.py
p4:
	python3 ./src/pipeline/D_train.py
p5:
	python3 ./src/pipeline/E_evaluate.py
```

old repo: 
https://mygit.th-deg.de/ml-wlan/predinet