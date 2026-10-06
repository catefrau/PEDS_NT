model_names = ["PEDS_gauss", "MLP"]

curves = []

for m in model_names:

    file = f"data/training_results/{m}/training_curves.npz"

    # read file and get test curves, append it


# plot them all together against number of epochs