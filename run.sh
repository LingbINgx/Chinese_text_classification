
source .venv/bin/activate

cd train

python scratch_transformer_trainer.py

python textcnn_trainer.py

python textrnn_trainer.py

python transformer_trainer.py