
### CORA PUBLIC SPLIT ###
# Baseline
python gcn.py --dataset=Cora --split=public --optimizer=Adam --name=BaselineGCNCoraWithFeatures
python gcn.py --dataset=Cora --split=public --optimizer=Adam --use_sparse=True --model=SparseNet --name=SparseBaselineGCNCoraWithFeatures

# Run Cora with dense
python gcn.py --dataset=Cora --split=public --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=1000 --params_nnz=10 --depth=0 --name=RIDenseGCNDepth0WithFeatures
python gcn.py --dataset=Cora --split=public --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=1000 --params_nnz=10 --depth=1 --name=RIDenseGCNDepth1WithFeatures
python gcn.py --dataset=Cora --split=public --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=1000 --params_nnz=10 --depth=2 --name=RIDenseGCNDepth2WithFeatures
python gcn.py --dataset=Cora --split=public --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=1000 --params_nnz=10 --depth=3 --name=RIDenseGCNDepth3WithFeatures

# Run Cora with sparse
python gcn.py --dataset=Cora --split=public --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=1000 --params_nnz=10 --depth=0 --use_sparse=True --name=RISparseGCNDepth0WithFeatures
python gcn.py --dataset=Cora --split=public --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=1000 --params_nnz=10 --depth=1 --use_sparse=True --name=RISparseGCNDepth1WithFeatures
python gcn.py --dataset=Cora --split=public --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=1000 --params_nnz=10 --depth=2 --use_sparse=True --name=RISparseGCNDepth2WithFeatures
python gcn.py --dataset=Cora --split=public --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=1000 --params_nnz=10 --depth=3 --use_sparse=True --name=RISparseGCNDepth3WithFeatures


# Baseline
python3 gcn.py --dataset=Cora --split=public --features_as=OneHotNodes --name=BaselineGCNCoraWithoutFeatures
# SparseNet baseline
python3 gcn.py --dataset=Cora --split=public --use_sparse=True --model=SparseNet --features_as=OneHotNodes --name=SparseBaselineGCNCoraWithoutFeatures

# Without node features, dense GCN
python3 gcn.py --dataset=Cora --split=public --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1000 --params_nnz=10 --depth=0 --name=RIDenseGCNDepth0WithFeatures
python3 gcn.py --dataset=Cora --split=public --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1000 --params_nnz=10 --depth=1 --name=RIDenseGCNDepth1WithFeatures
python3 gcn.py --dataset=Cora --split=public --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1000 --params_nnz=10 --depth=2 --name=RIDenseGCNDepth2WithFeatures
python3 gcn.py --dataset=Cora --split=public --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1000 --params_nnz=10 --depth=3 --name=RIDenseGCNDepth3WithFeatures


python3 gcn.py --dataset=Cora --split=public --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1000 --params_nnz=10 --depth=0 --use_sparse=True --name=RISparseGCNDepth0WithFeatures
python3 gcn.py --dataset=Cora --split=public --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1000 --params_nnz=10 --depth=1 --use_sparse=True --name=RISparseGCNDepth1WithFeatures
python3 gcn.py --dataset=Cora --split=public --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1000 --params_nnz=10 --depth=2 --use_sparse=True --name=RISparseGCNDepth2WithFeatures
python3 gcn.py --dataset=Cora --split=public --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1000 --params_nnz=10 --depth=3 --use_sparse=True --name=RISparseGCNDepth3WithFeatures


### CORA complete ###
# Baseline
python gcn.py --dataset=Cora --split=complete --optimizer=Adam --name=BaselineGCNCoraWithFeatures
# SparseNet baseline
python gcn.py --dataset=Cora --split=complete --optimizer=Adam --use_sparse=True --model=SparseNet --name=SparseBaselineGCNCoraWithFeatures

# Run Cora with dense
python gcn.py --dataset=Cora --split=complete --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=1000 --params_nnz=10 --depth=0 --name=RIDenseGCNDepth0WithFeatures
python gcn.py --dataset=Cora --split=complete --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=1000 --params_nnz=10 --depth=1 --name=RIDenseGCNDepth1WithFeatures
python gcn.py --dataset=Cora --split=complete --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=1000 --params_nnz=10 --depth=2 --name=RIDenseGCNDepth2WithFeatures
python gcn.py --dataset=Cora --split=complete --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=1000 --params_nnz=10 --depth=3 --name=RIDenseGCNDepth3WithFeatures

# Run Cora with sparse
python gcn.py --dataset=Cora --split=complete --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=1000 --params_nnz=10 --depth=0 --use_sparse=True--name=RISparseGCNDepth0WithFeatures
python gcn.py --dataset=Cora --split=complete --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=1000 --params_nnz=10 --depth=1 --use_sparse=True--name=RISparseGCNDepth1WithFeatures
python gcn.py --dataset=Cora --split=complete --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=1000 --params_nnz=10 --depth=2 --use_sparse=True--name=RISparseGCNDepth2WithFeatures
python gcn.py --dataset=Cora --split=complete --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=1000 --params_nnz=10 --depth=3 --use_sparse=True--name=RISparseGCNDepth3WithFeatures


# Baseline
python3 gcn.py --dataset=Cora --split=complete --features_as=OneHotNodes --name=BaselineGCNCoraWithoutFeatures
# SparseNet baseline
python3 gcn.py --dataset=Cora --split=complete --use_sparse=True --model=SparseNet --features_as=OneHotNodes --name=SparseBaselineGCNCoraWithoutFeatures

# Without node features, dense GCN
python3 gcn.py --dataset=Cora --split=complete --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1500 --params_nnz=10 --depth=0 --name=RIDenseGCNDepth0WithoutFeatures;
python3 gcn.py --dataset=Cora --split=complete --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1500 --params_nnz=10 --depth=1 --name=RIDenseGCNDepth1WithoutFeatures;
python3 gcn.py --dataset=Cora --split=complete --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1500 --params_nnz=10 --depth=2 --name=RIDenseGCNDepth2WithoutFeatures;
python3 gcn.py --dataset=Cora --split=complete --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1500 --params_nnz=10 --depth=3 --name=RIDenseGCNDepth3WithoutFeatures;


python3 gcn.py --dataset=Cora --split=complete --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1500 --params_nnz=10 --depth=0 --use_sparse=True --name=RISparseGCNDepth0WithoutFeatures;
python3 gcn.py --dataset=Cora --split=complete --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1500 --params_nnz=10 --depth=1 --use_sparse=True --name=RISparseGCNDepth1WithoutFeatures;
python3 gcn.py --dataset=Cora --split=complete --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1500 --params_nnz=10 --depth=2 --use_sparse=True --name=RISparseGCNDepth2WithoutFeatures;
python3 gcn.py --dataset=Cora --split=complete --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1500 --params_nnz=10 --depth=3 --use_sparse=True --name=RISparseGCNDepth3WithoutFeatures

# Nice results here
python3 gcn.py --dataset=Cora --split=complete --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1500 --params_nnz=10 --depth=1 --use_sparse=True --name=RISparseGCNDepth1WithoutFeatures;

### CiteSeer PUBLIC SPLIT ###
# Baseline
python gcn.py --dataset=CiteSeer --split=public --optimizer=Adam --name=BaselineGCNCiteSeerWithFeatures;
python gcn.py --dataset=CiteSeer --split=public --optimizer=Adam --use_sparse=True --model=SparseNet --name=SparseBaselineGCNCiteSeerWithFeatures

# Run CiteSeer with dense
python gcn.py --dataset=CiteSeer --split=public --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=2000 --params_nnz=2 --depth=0 --name=RIDenseGCNDepth0WithFeatures;
python gcn.py --dataset=CiteSeer --split=public --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=2000 --params_nnz=2 --depth=1 --name=RIDenseGCNDepth1WithFeatures
python gcn.py --dataset=CiteSeer --split=public --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=2000 --params_nnz=2 --depth=2 --name=RIDenseGCNDepth2WithFeatures
python gcn.py --dataset=CiteSeer --split=public --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=2000 --params_nnz=2 --depth=3 --name=RIDenseGCNDepth3WithFeatures

# Run CiteSeer with sparse
python gcn.py --dataset=CiteSeer --split=public --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=2000 --params_nnz=2 --depth=0 --use_sparse=True;
python gcn.py --dataset=CiteSeer --split=public --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=2000 --params_nnz=2 --depth=1 --use_sparse=True;
python gcn.py --dataset=CiteSeer --split=public --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=2000 --params_nnz=2 --depth=2 --use_sparse=True;
python gcn.py --dataset=CiteSeer --split=public --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=2000 --params_nnz=2 --depth=3 --use_sparse=True


# Baseline
python3 gcn.py --dataset=CiteSeer --split=public --features_as=OneHotNodes --name=BaselineGCNCiteSeerWithoutFeatures;
python3 gcn.py --dataset=CiteSeer --split=public --use_sparse=True --model=SparseNet --features_as=OneHotNodes --name=SparseBaselineGCNCiteSeerWithoutFeatures

# Without node features, dense GCN
python3 gcn.py --dataset=CiteSeer --split=public --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1000 --params_nnz=2 --depth=0 --name=RIDenseGCNDepth0WithoutFeatures;
python3 gcn.py --dataset=CiteSeer --split=public --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1000 --params_nnz=2 --depth=1 --name=RIDenseGCNDepth1WithoutFeatures;
python3 gcn.py --dataset=CiteSeer --split=public --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1000 --params_nnz=2 --depth=2 --name=RIDenseGCNDepth2WithoutFeatures;
python3 gcn.py --dataset=CiteSeer --split=public --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1000 --params_nnz=2 --depth=3 --name=RIDenseGCNDepth3WithoutFeatures;


python3 gcn.py --dataset=CiteSeer --split=public --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1000 --params_nnz=2 --depth=0 --use_sparse=True --name=RISparseGCNDepth0WithoutFeatures;
python3 gcn.py --dataset=CiteSeer --split=public --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1000 --params_nnz=2 --depth=1 --use_sparse=True --name=RISparseGCNDepth1WithoutFeatures;
python3 gcn.py --dataset=CiteSeer --split=public --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1000 --params_nnz=2 --depth=2 --use_sparse=True --name=RISparseGCNDepth2WithoutFeatures;
python3 gcn.py --dataset=CiteSeer --split=public --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1000 --params_nnz=2 --depth=3 --use_sparse=True --name=RISparseGCNDepth3WithoutFeatures


### CiteSeer complete ###
# Baseline
python gcn.py --dataset=CiteSeer --split=complete --optimizer=Adam --name=BaselineGCNCiteSeerComplete
# SparseNet baseline
python gcn.py --dataset=CiteSeer --split=complete --optimizer=Adam --use_sparse=True --model=SparseNet --name=SparseBaselineGCNCiteSeerComplete

# Run CiteSeer with dense
python gcn.py --dataset=CiteSeer --split=complete --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=2000 --params_nnz=10 --depth=0 --name=RIDenseGCNDepth0WithFeatures
python gcn.py --dataset=CiteSeer --split=complete --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=2000 --params_nnz=10 --depth=1 --name=RIDenseGCNDepth1WithFeatures
python gcn.py --dataset=CiteSeer --split=complete --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=2000 --params_nnz=10 --depth=2 --name=RIDenseGCNDepth2WithFeatures
python gcn.py --dataset=CiteSeer --split=complete --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=2000 --params_nnz=6 --depth=3 --name=RIDenseGCNDepth3WithFeatures

# Run CiteSeer with sparse
python gcn.py --dataset=CiteSeer --split=complete --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=2000 --params_nnz=10 --depth=0 --use_sparse=True--name=RISparseGCNDepth0WithFeatures
python gcn.py --dataset=CiteSeer --split=complete --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=2000 --params_nnz=10 --depth=1 --use_sparse=True--name=RISparseGCNDepth1WithFeatures
python gcn.py --dataset=CiteSeer --split=complete --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=2000 --params_nnz=10 --depth=2 --use_sparse=True--name=RISparseGCNDepth2WithFeatures
python gcn.py --dataset=CiteSeer --split=complete --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=2000 --params_nnz=10 --depth=3 --use_sparse=True--name=RISparseGCNDepth3WithFeatures


# Baseline
python3 gcn.py --dataset=CiteSeer --split=complete --features_as=OneHotNodes --name=BaselineGCNCiteSeerCompleteWithoutFeatures
# SparseNet baseline
python3 gcn.py --dataset=CiteSeer --split=complete --use_sparse=True --model=SparseNet --features_as=OneHotNodes --name=SparseBaselineGCNCiteSeerCompleteWithoutFeatures

# Without node features, dense GCN
python3 gcn.py --dataset=CiteSeer --split=complete --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1500 --params_nnz=6 --depth=0 --name=RIDenseGCNDepth0WithoutFeatures;
python3 gcn.py --dataset=CiteSeer --split=complete --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1500 --params_nnz=6 --depth=1 --name=RIDenseGCNDepth1WithoutFeatures;
python3 gcn.py --dataset=CiteSeer --split=complete --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1500 --params_nnz=6 --depth=2 --name=RIDenseGCNDepth2WithoutFeatures;
python3 gcn.py --dataset=CiteSeer --split=complete --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1500 --params_nnz=6 --depth=3 --name=RIDenseGCNDepth3WithoutFeatures;


python3 gcn.py --dataset=CiteSeer --split=complete --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1500 --params_nnz=6 --depth=0 --use_sparse=True --name=RISparseGCNDepth0WithoutFeatures;
python3 gcn.py --dataset=CiteSeer --split=complete --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=3000 --params_nnz=10 --depth=1 --use_sparse=True --name=RISparseGCNDepth1WithoutFeatures;
python3 gcn.py --dataset=CiteSeer --split=complete --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1500 --params_nnz=10 --depth=2 --use_sparse=True --name=RISparseGCNDepth2WithoutFeatures;
python3 gcn.py --dataset=CiteSeer --split=complete --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1500 --params_nnz=6 --depth=3 --use_sparse=True --name=RISparseGCNDepth3WithoutFeatures


### PubMed public ###
# Baseline
python gcn.py --dataset=PubMed --split=public --optimizer=Adam --name=BaselineGCNPubMedPublicWithFeatures
# SparseNet baseline
python gcn.py --dataset=PubMed --split=public --optimizer=Adam --use_sparse=True --model=SparseNet --name=SparseBaselineGCNPubMedPublicWithFeatures

# Run PubMed with dense
python gcn.py --dataset=PubMed --split=public --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=1500 --params_nnz=6 --depth=0 --name=RIDenseGCNDepth0WithFeatures
python gcn.py --dataset=PubMed --split=public --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=1500 --params_nnz=6 --depth=1 --name=RIDenseGCNDepth1WithFeatures
python gcn.py --dataset=PubMed --split=public --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=1500 --params_nnz=6 --depth=2 --name=RIDenseGCNDepth2WithFeatures
python gcn.py --dataset=PubMed --split=public --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=1500 --params_nnz=6 --depth=3 --name=RIDenseGCNDepth3WithFeatures

# Run PubMed with sparse
python gcn.py --dataset=PubMed --split=public --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=400 --params_nnz=4 --depth=0 --use_sparse=True--name=RISparseGCNDepth0WithFeatures
python gcn.py --dataset=PubMed --split=public --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=400 --params_nnz=4 --depth=1 --use_sparse=True--name=RISparseGCNDepth1WithFeatures
python gcn.py --dataset=PubMed --split=public --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=400 --params_nnz=4 --depth=2 --use_sparse=True--name=RISparseGCNDepth2WithFeatures
python gcn.py --dataset=PubMed --split=public --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=400 --params_nnz=4 --depth=3 --use_sparse=True--name=RISparseGCNDepth3WithFeatures


# Baseline
python3 gcn.py --dataset=PubMed --split=public --features_as=OneHotNodes --name=BaselineGCNPubMedPublicWithoutFeatures
# SparseNet baseline
python3 gcn.py --dataset=PubMed --split=public --use_sparse=True --model=SparseNet --features_as=OneHotNodes --name=SparseBaselineGCNPubMedPublicWithoutFeatures

# Without node features, dense GCN
python3 gcn.py --dataset=PubMed --split=public --features_as=IndexVecs --model=RINet --params_features_as=initialization_as_context --params_dim=1000 --params_nnz=6 --depth=0 --name=RIDenseGCNDepth0WithoutFeatures;
python3 gcn.py --dataset=PubMed --split=public --features_as=IndexVecs --model=RINet --params_features_as=initialization_as_context --params_dim=1000 --params_nnz=6 --depth=1 --name=RIDenseGCNDepth1WithoutFeatures;
python3 gcn.py --dataset=PubMed --split=public --features_as=IndexVecs --model=RINet --params_features_as=initialization_as_context --params_dim=1000 --params_nnz=6 --depth=2 --name=RIDenseGCNDepth2WithoutFeatures;
python3 gcn.py --dataset=PubMed --split=public --features_as=IndexVecs --model=RINet --params_features_as=initialization_as_context --params_dim=1000 --params_nnz=6 --depth=3 --name=RIDenseGCNDepth3WithoutFeatures


python3 gcn.py --dataset=PubMed --split=public --features_as=IndexVecs --model=RINet --params_features_as=initialization_as_context --params_dim=1000 --params_nnz=6 --use_sparse=True --depth=0 --name=RIDenseGCNDepth0WithoutFeatures;
python3 gcn.py --dataset=PubMed --split=public --features_as=IndexVecs --model=RINet --params_features_as=initialization_as_context --params_dim=1000 --params_nnz=6 --use_sparse=True --depth=1 --name=RIDenseGCNDepth1WithoutFeatures;
python3 gcn.py --dataset=PubMed --split=public --features_as=IndexVecs --model=RINet --params_features_as=initialization_as_context --params_dim=1000 --params_nnz=6 --use_sparse=True --depth=2 --name=RIDenseGCNDepth2WithoutFeatures;
python3 gcn.py --dataset=PubMed --split=public --features_as=IndexVecs --model=RINet --params_features_as=initialization_as_context --params_dim=1000 --params_nnz=6 --use_sparse=True --depth=3 --name=RIDenseGCNDepth3WithoutFeatures


python3 gcn.py --dataset=PubMed --split=public --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=10000 --params_nnz=10 --depth=0 --use_sparse=True --name=RISparseGCNDepth0WithoutFeatures
python3 gcn.py --dataset=PubMed --split=public --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=10000 --params_nnz=10 --depth=1 --use_sparse=True --name=RISparseGCNDepth1WithoutFeatures
python3 gcn.py --dataset=PubMed --split=public --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=10000 --params_nnz=10 --depth=2 --use_sparse=True --name=RISparseGCNDepth2WithoutFeatures
python3 gcn.py --dataset=PubMed --split=public --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=10000 --params_nnz=10 --depth=3 --use_sparse=True --name=RISparseGCNDepth3WithoutFeatures


### PubMed complete ###
# Baseline
python gcn.py --dataset=PubMed --split=complete --optimizer=Adam --name=BaselineGCNPubMedCompleteWithFeatures
# SparseNet baseline
python gcn.py --dataset=PubMed --split=complete --optimizer=Adam --use_sparse=True --model=SparseNet --name=SparseBaselineGCNPubMedCompleteWithFeatures

# Run PubMed with dense
python gcn.py --dataset=PubMed --split=complete --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=400 --params_nnz=2 --depth=0 --name=RIDenseGCNDepth0WithFeatures;
python gcn.py --dataset=PubMed --split=complete --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=400 --params_nnz=2 --depth=1 --name=RIDenseGCNDepth1WithFeatures;
python gcn.py --dataset=PubMed --split=complete --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=1000 --params_nnz=6 --depth=2 --name=RIDenseGCNDepth2WithFeatures;
python gcn.py --dataset=PubMed --split=complete --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=1000 --params_nnz=6 --depth=3 --name=RIDenseGCNDepth3WithFeatures

# Run PubMed with sparse
python gcn.py --dataset=PubMed --split=public --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=400 --params_nnz=6 --depth=0 --use_sparse=True--name=RISparseGCNDepth0WithFeatures;
python gcn.py --dataset=PubMed --split=public --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=250 --params_nnz=2 --depth=1 --use_sparse=True--name=RISparseGCNDepth1WithFeatures;
python gcn.py --dataset=PubMed --split=public --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=250 --params_nnz=6 --depth=2 --use_sparse=True--name=RISparseGCNDepth2WithFeatures;
python gcn.py --dataset=PubMed --split=public --optimizer=Adam --model="RINet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=250 --params_nnz=6 --depth=3 --use_sparse=True--name=RISparseGCNDepth3WithFeatures


# Baseline
python3 gcn.py --dataset=PubMed --split=complete --features_as=OneHotNodes --name=BaselineGCNPubMedCompleteWithoutFeatures
# SparseNet baseline
python3 gcn.py --dataset=PubMed --split=complete --use_sparse=True --model=SparseNet --features_as=OneHotNodes --name=SparseBaselineGCNPubMedCompleteWithoutFeatures

# Without node features, dense GCN
python3 gcn.py --dataset=PubMed --split=complete --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=10000 --params_nnz=10 --depth=0 --name=RIDenseGCNDepth0WithoutFeatures
python3 gcn.py --dataset=PubMed --split=complete --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=10000 --params_nnz=10 --depth=1 --name=RIDenseGCNDepth1WithoutFeatures
python3 gcn.py --dataset=PubMed --split=complete --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=10000 --params_nnz=10 --depth=2 --name=RIDenseGCNDepth2WithoutFeatures
python3 gcn.py --dataset=PubMed --split=complete --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=10000 --params_nnz=10 --depth=3 --name=RIDenseGCNDepth3WithoutFeatures


python3 gcn.py --dataset=PubMed --split=complete --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1500 --params_nnz=10 --depth=0 --use_sparse=True --name=RISparseGCNDepth0WithoutFeatures;
python3 gcn.py --dataset=PubMed --split=complete --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1500 --params_nnz=10 --depth=1 --use_sparse=True --name=RISparseGCNDepth1WithoutFeatures;
python3 gcn.py --dataset=PubMed --split=complete --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1500 --params_nnz=10 --depth=2 --use_sparse=True --name=RISparseGCNDepth2WithoutFeatures;
python3 gcn.py --dataset=PubMed --split=complete --features_as=IndexVecs --model=RINet --params_features_as=excluded --params_dim=1500 --params_nnz=10 --depth=3 --use_sparse=True --name=RISparseGCNDepth3WithoutFeatures
