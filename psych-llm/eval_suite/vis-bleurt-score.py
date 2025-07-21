import pickle
import matplotlib.pyplot as plt

def main(scorefile: str = None, expt_path: str = None) -> None:

    if scorefile is None: raise Exception("No score file found.")
    else:
        with open(scorefile, 'rb') as f:
            scores = pickle.load(f)
        fig, ax = plt.subplots()
        ax.hist(scores,bins=100)
        ax.set(xlabel="BLEURT score", ylabel="Frequency")
        ax.set_title("Distribution of BLEURT scores over 5k samples")
        if expt_path:
            fig.savefig(f'{expt_path}bleurtscores.png', dpi=300)
        else: fig.savefig('bleurtscores.png')

if __name__ == "__main__":
    expt_path = './../n0.50_r0.50/'
    scorefile = f'{expt_path}scores.pkl'
    main(scorefile, expt_path)
