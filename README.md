# SynChemTS

MCTS-Based De Novo Molecular Design with Ligand-Efficiency Feedback

A structure-based de novo molecular design framework built on Monte Carlo Tree Search (MCTS). It improves on earlier MCTS + docking pipelines by (1) replacing the raw docking score with a size-compensated ligand-efficiency reward, (2) periodically re-injecting promising molecules into the search tree through synonymous SMILES transformations, and (3) using two complementary generative models: one for standard SMILES and one for non-standard SMILES.

Background and Motivation

Earlier de novo design approaches that combine MCTS with a generative SMILES model have two main limitations.

1. Using the docking score directly as the reward is biased toward large molecules. Docking scores tend to improve (become more negative) as molecular weight increases, so larger molecules obtain a lower average binding energy almost regardless of quality. If the docking score is used directly as the reward, the search stops rewarding the contribution of each individual atom and instead drifts toward ever-longer molecules in order to collect a better score.

2. The linear nature of SMILES restricts the optimization direction. SMILES strings are generated token by token, so optimization is effectively linear. Once part of a molecule has been built, there is no opportunity to go back and revise it. Good molecules are therefore only refined "forward", never globally.

Key Contributions
1. Ligand Efficiency (LE) combined with docking score, with a size-compensation coefficient
The reward combines LE and the docking score, so that the search favors molecules whose atoms contribute efficiently to binding rather than molecules that are simply large.
LE naturally decreases as molecules grow, because larger molecules need linkers to join their high-contribution fragments, and linkers contribute little to binding. To avoid penalizing legitimately larger molecules, a compensation coefficient is applied to LE as a function of molecular size.
A purely linear compensation would reward unbounded growth and bring back the length-inflation problem. The compensation coefficient is therefore a sigmoid function of molecular size, which gives it a fixed upper bound: larger molecules receive compensation, but only up to a limit.
2. Synonymous SMILES transformation and re-insertion into the search tree
At user-definable time points during the search, promising molecules are selected and converted into synonymous (equivalent) SMILES representations of the same molecule.
These transformed strings are re-inserted into the MCTS tree, which changes the direction of optimization and gives the search a chance to revise parts of a molecule that the linear generation order would otherwise lock in.
Naively adding these molecules token by token would create many new branches, wasting resources and hindering convergence. Following the approach of the --input option in ChemTSv2, the entire SMILES string is treated as a single node, which avoids the branch explosion.
3. Dual-model architecture
Model	Purpose
Standard SMILES model	Finds basic scaffolds and performs the initial optimization
Non-standard SMILES model	Optimizes molecules written as non-standard (non-canonical) SMILES, such as those produced by the synonymous transformation step


Reward Function

The reward is built from the docking score and ligand efficiency, with a sigmoid-bounded size compensation applied to LE:

Docking score: binding score from the docking program.
LE: docking score normalized by molecular size (e.g., heavy atom count).
Compensation coefficient: c(N) = sigmoid(N)-shaped, monotonically increasing in molecular size N and bounded above, so large molecules are compensated but cannot grow without limit.

The exact functional form, normalization, and hyperparameters (sigmoid midpoint, steepness, upper bound, weighting between LE and docking score) are defined in the source code. See the configuration files for the values used in each experiment.
