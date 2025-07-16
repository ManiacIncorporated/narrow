import pickle
import matplotlib.pyplot as plt

def main(scorefile: str = None) -> None:

    if scorefile is None: raise Exception("No score file found.")
    else:
        with open(scorefile, 'rb') as f:
            scores = pickle.load(f)
        fig, ax = plt.subplots()
        ax.hist(scores)
        ax.set(xlabel="BLEURT score", ylabel="Frequency")
        ax.set_title("Distribution of BLEURT scores over 5k samples")
        fig.savefig('bleurtscores.png', dpi=300)

if __name__ == "__main__":
    scorefile = 'scores.pkl'
    main(scorefile)
