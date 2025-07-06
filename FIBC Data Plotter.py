# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 15:30:41 2025
Code to plot the overall specific charge drop data for polypropylene Faraday cup drops from lined and unlined FIBCs
@author: vb22224
"""

import pandas as pd
import matplotlib.pyplot as plt
import os



if __name__ == "__main__":
    
    path = "C:\\Users\\vb22224\\OneDrive - University of Bristol\\Desktop\\MAIN\\Data for IEEE Paper\\FIBC Charging\\FIBC Data.csv"

    df = pd.read_csv(path)  # Load data

    if 'Lined' in df.columns and 'Unlined' in df.columns:  # Check columns exist
        
        # Define output filename based on same directory as CSV
        out_dir = os.path.dirname(path)
        out_file = os.path.join(out_dir, "FIBC_boxplot.pdf")
        
        plt.figure(dpi=600, figsize=(6, 3))
        ax = df[['Lined', 'Unlined']].boxplot(vert=False)  # Horizontal boxplot
        ax.grid(False)
        plt.xlabel('Specific Charge / C kg$^{-1}$')  # Update axis label
        plt.tight_layout()
        plt.savefig(out_file)  # Save before showing
        plt.show()
        
    else:
        print("Columns 'Lined' and/or 'Unlined' not found in the CSV.")

