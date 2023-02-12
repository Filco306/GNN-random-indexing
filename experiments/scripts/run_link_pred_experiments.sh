
### CORA PUBLIC SPLIT ###
# Baseline
python model_linkpred.py --dataset=Cora --optimizer=Adam --name=Baselinemodel_linkpredCoraWithFeatures
python model_linkpred.py --dataset=Cora --optimizer=Adam --name=Baselinemodel_linkpredCoraWithFeatures --model=RILinkPredNet --depth=1
# SparseNet baseline
python model_linkpred.py --dataset=Cora --optimizer=Adam --use_sparse=True --name=SparseBaselinemodel_linkpredCoraWithFeatures
python model_linkpred.py --dataset=Cora --optimizer=Adam --use_sparse=True --name=SparseBaselinemodel_linkpredCoraWithFeatures --model=RILinkPredNet --depth=1

# Run Cora with dense
python model_linkpred.py --dataset=Cora --optimizer=Adam --model="RILinkPredNet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=1000 --params_nnz=2 --depth=0 --name=RIDensemodel_linkpredDepth0WithFeatures
python model_linkpred.py --dataset=Cora --optimizer=Adam --model="RILinkPredNet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=1000 --params_nnz=2 --depth=1 --name=RIDensemodel_linkpredDepth1WithFeatures
python model_linkpred.py --dataset=Cora --optimizer=Adam --model="RILinkPredNet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=1000 --params_nnz=2 --depth=2 --name=RIDensemodel_linkpredDepth2WithFeatures
python model_linkpred.py --dataset=Cora --optimizer=Adam --model="RILinkPredNet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=1000 --params_nnz=2 --depth=3 --name=RIDensemodel_linkpredDepth3WithFeatures

# Run Cora with sparse
python model_linkpred.py --dataset=Cora --optimizer=Adam --model="RILinkPredNet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=1000 --params_nnz=2 --depth=0 --use_sparse=True --name=RISparsemodel_linkpredDepth0WithFeatures
python model_linkpred.py --dataset=Cora --optimizer=Adam --model="RILinkPredNet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=1000 --params_nnz=2 --depth=1 --use_sparse=True --name=RISparsemodel_linkpredDepth1WithFeatures
python model_linkpred.py --dataset=Cora --optimizer=Adam --model="RILinkPredNet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=1000 --params_nnz=2 --depth=2 --use_sparse=True --name=RISparsemodel_linkpredDepth2WithFeatures
python model_linkpred.py --dataset=Cora --optimizer=Adam --model="RILinkPredNet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=1000 --params_nnz=2 --depth=3 --use_sparse=True --name=RISparsemodel_linkpredDepth3WithFeatures


# Baseline
python3 model_linkpred.py --dataset=Cora --features_as=OneHotNodes --name=Baselinemodel_linkpredCoraWithoutFeatures

# Baseline adjacency matrix
python3 model_linkpred.py --dataset=Cora --features_as=OneHotNodes --name=Baselinemodel_linkpredCoraWithoutFeatures --model=RILinkPredNet --depth=1
python3 model_linkpred.py --dataset=Cora --use_sparse=True --features_as=OneHotNodes --name=SparseBaselinemodel_linkpredCoraWithoutFeatures
python3 model_linkpred.py --dataset=Cora --use_sparse=True --features_as=OneHotNodes --name=SparseBaselinemodel_linkpredCoraWithoutFeatures--model=RILinkPredNet --depth=1

# Without node features, dense model_linkpred
python3 model_linkpred.py --dataset=Cora --features_as=IndexVecs --model=RILinkPredNet --params_features_as=excluded --params_dim=1000 --params_nnz=10 --depth=0 --name=RIDensemodel_linkpredDepth0WithFeatures
python3 model_linkpred.py --dataset=Cora --features_as=IndexVecs --model=RILinkPredNet --params_features_as=excluded --params_dim=1000 --params_nnz=10 --depth=1 --name=RIDensemodel_linkpredDepth1WithFeatures
python3 model_linkpred.py --dataset=Cora --features_as=IndexVecs --model=RILinkPredNet --params_features_as=excluded --params_dim=1000 --params_nnz=10 --depth=2 --name=RIDensemodel_linkpredDepth2WithFeatures
python3 model_linkpred.py --dataset=Cora --features_as=IndexVecs --model=RILinkPredNet --params_features_as=excluded --params_dim=1000 --params_nnz=10 --depth=3 --name=RIDensemodel_linkpredDepth3WithFeatures


python3 model_linkpred.py --dataset=Cora --features_as=IndexVecs --model=RILinkPredNet --params_features_as=excluded --params_dim=1000 --params_nnz=10 --depth=0 --use_sparse=True --name=RISparsemodel_linkpredDepth0WithFeatures
python3 model_linkpred.py --dataset=Cora --features_as=IndexVecs --model=RILinkPredNet --params_features_as=excluded --params_dim=1000 --params_nnz=10 --depth=1 --use_sparse=True --name=RISparsemodel_linkpredDepth1WithFeatures
python3 model_linkpred.py --dataset=Cora --features_as=IndexVecs --model=RILinkPredNet --params_features_as=excluded --params_dim=1000 --params_nnz=10 --depth=2 --use_sparse=True --name=RISparsemodel_linkpredDepth2WithFeatures
python3 model_linkpred.py --dataset=Cora --features_as=IndexVecs --model=RILinkPredNet --params_features_as=excluded --params_dim=1000 --params_nnz=10 --depth=3 --use_sparse=True --name=RISparsemodel_linkpredDepth3WithFeatures


### CiteSeer PUBLIC SPLIT ###
# Baseline
python model_linkpred.py --dataset=CiteSeer --optimizer=Adam --name=Baselinemodel_linkpredCiteSeerWithFeatures;
python model_linkpred.py --dataset=CiteSeer --optimizer=Adam --name=Baselinemodel_linkpredCiteSeerWithFeatures;--model=RILinkPredNet --depth=1
python model_linkpred.py --dataset=CiteSeer --optimizer=Adam --use_sparse=True --name=SparseBaselinemodel_linkpredCiteSeerWithFeatures
python model_linkpred.py --dataset=CiteSeer --optimizer=Adam --use_sparse=True --name=SparseBaselinemodel_linkpredCiteSeerWithFeatures--model=RILinkPredNet --depth=1

# Run CiteSeer with dense
python model_linkpred.py --dataset=CiteSeer --optimizer=Adam --model="RILinkPredNet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=2000 --params_nnz=2 --depth=0;
python model_linkpred.py --dataset=CiteSeer --optimizer=Adam --model="RILinkPredNet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=2000 --params_nnz=2 --depth=1;
python model_linkpred.py --dataset=CiteSeer --optimizer=Adam --model="RILinkPredNet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=2000 --params_nnz=2 --depth=2;
python model_linkpred.py --dataset=CiteSeer --optimizer=Adam --model="RILinkPredNet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=2000 --params_nnz=2 --depth=3;

# Run CiteSeer with sparse
python model_linkpred.py --dataset=CiteSeer --optimizer=Adam --model="RILinkPredNet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=2000 --params_nnz=2 --depth=0 --use_sparse=True;
python model_linkpred.py --dataset=CiteSeer --optimizer=Adam --model="RILinkPredNet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=2000 --params_nnz=2 --depth=1 --use_sparse=True;
python model_linkpred.py --dataset=CiteSeer --optimizer=Adam --model="RILinkPredNet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=2000 --params_nnz=2 --depth=2 --use_sparse=True;
python model_linkpred.py --dataset=CiteSeer --optimizer=Adam --model="RILinkPredNet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=2000 --params_nnz=2 --depth=3 --use_sparse=True


# Baseline
python3 model_linkpred.py --dataset=CiteSeer --features_as=OneHotNodes --name=Baselinemodel_linkpredCiteSeerWithoutFeatures;
python3 model_linkpred.py --dataset=CiteSeer --features_as=OneHotNodes --name=Baselinemodel_linkpredCiteSeerWithoutFeatures --model=RILinkPredNet --depth=1
python3 model_linkpred.py --dataset=CiteSeer --use_sparse=True --features_as=OneHotNodes --name=SparseBaselinemodel_linkpredCiteSeerWithoutFeatures
python3 model_linkpred.py --dataset=CiteSeer --use_sparse=True --features_as=OneHotNodes --name=SparseBaselinemodel_linkpredCiteSeerWithoutFeatures --model=RILinkPredNet --depth=1


# Without node features, dense model_linkpred
python3 model_linkpred.py --dataset=CiteSeer --features_as=IndexVecs --model=RILinkPredNet --params_features_as=excluded --params_dim=1000 --params_nnz=10 --depth=0 --name=RIDensemodel_linkpredDepth0WithoutFeatures
python3 model_linkpred.py --dataset=CiteSeer --features_as=IndexVecs --model=RILinkPredNet --params_features_as=excluded --params_dim=1000 --params_nnz=10 --depth=1 --name=RIDensemodel_linkpredDepth1WithoutFeatures
python3 model_linkpred.py --dataset=CiteSeer --features_as=IndexVecs --model=RILinkPredNet --params_features_as=excluded --params_dim=1000 --params_nnz=10 --depth=2 --name=RIDensemodel_linkpredDepth2WithoutFeatures
python3 model_linkpred.py --dataset=CiteSeer --features_as=IndexVecs --model=RILinkPredNet --params_features_as=excluded --params_dim=1000 --params_nnz=10 --depth=3 --name=RIDensemodel_linkpredDepth3WithoutFeatures


python3 model_linkpred.py --dataset=CiteSeer --features_as=IndexVecs --model=RILinkPredNet --params_features_as=excluded --params_dim=1000 --params_nnz=10 --depth=0 --use_sparse=True --name=RISparsemodel_linkpredDepth0WithoutFeatures
python3 model_linkpred.py --dataset=CiteSeer --features_as=IndexVecs --model=RILinkPredNet --params_features_as=excluded --params_dim=1000 --params_nnz=10 --depth=1 --use_sparse=True --name=RISparsemodel_linkpredDepth1WithoutFeatures
python3 model_linkpred.py --dataset=CiteSeer --features_as=IndexVecs --model=RILinkPredNet --params_features_as=excluded --params_dim=1000 --params_nnz=10 --depth=2 --use_sparse=True --name=RISparsemodel_linkpredDepth2WithoutFeatures
python3 model_linkpred.py --dataset=CiteSeer --features_as=IndexVecs --model=RILinkPredNet --params_features_as=excluded --params_dim=1000 --params_nnz=10 --depth=3 --use_sparse=True --name=RISparsemodel_linkpredDepth3WithoutFeatures



### PubMed public ###
# Baseline
python model_linkpred.py --dataset=BlogCatalog --optimizer=Adam --name=Baselinemodel_linkpredPubMedPublicWithFeatures --model=RILinkPredNet --depth=0
python model_linkpred.py --dataset=BlogCatalog --optimizer=Adam --name=Baselinemodel_linkpredPubMedPublicWithFeatures --model=RILinkPredNet --depth=1
python model_linkpred.py --dataset=BlogCatalog --optimizer=Adam --name=Baselinemodel_linkpredPubMedPublicWithFeatures --model=RILinkPredNet --depth=2
python model_linkpred.py --dataset=BlogCatalog --optimizer=Adam --name=Baselinemodel_linkpredPubMedPublicWithFeatures --model=RILinkPredNet --depth=3
# SparseNet baseline
python model_linkpred.py --dataset=BlogCatalog --optimizer=Adam --use_sparse=True --name=SparseBaselinemodel_linkpredPubMedPublicWithFeatures --model=RILinkPredNet --depth=0
python model_linkpred.py --dataset=BlogCatalog --optimizer=Adam --name=Baselinemodel_linkpredPubMedPublicWithFeatures --model=RILinkPredNet --depth=1 --use_sparse=True
python model_linkpred.py --dataset=BlogCatalog --optimizer=Adam --name=Baselinemodel_linkpredPubMedPublicWithFeatures --model=RILinkPredNet --depth=2 --use_sparse=True
python model_linkpred.py --dataset=BlogCatalog --optimizer=Adam --name=Baselinemodel_linkpredPubMedPublicWithFeatures --model=RILinkPredNet --depth=3 --use_sparse=True

# Run PubMed with dense
python model_linkpred.py --dataset=BlogCatalog --optimizer=Adam --model="RILinkPredNet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=5000 --params_nnz=2 --depth=0 --name=RIDensemodel_linkpredDepth0WithFeatures
python model_linkpred.py --dataset=BlogCatalog --optimizer=Adam --model="RILinkPredNet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=5000 --params_nnz=2 --depth=1 --name=RIDensemodel_linkpredDepth1WithFeatures
python model_linkpred.py --dataset=BlogCatalog --optimizer=Adam --model="RILinkPredNet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=5000 --params_nnz=2 --depth=2 --name=RIDensemodel_linkpredDepth2WithFeatures
python model_linkpred.py --dataset=BlogCatalog --optimizer=Adam --model="RILinkPredNet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=5000 --params_nnz=2 --depth=3 --name=RIDensemodel_linkpredDepth3WithFeatures

# Run PubMed with sparse
python model_linkpred.py --dataset=BlogCatalog --optimizer=Adam --model="RILinkPredNet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=5000 --params_nnz=2 --depth=0 --use_sparse=True--name=RISparsemodel_linkpredDepth0WithFeatures
python model_linkpred.py --dataset=BlogCatalog --optimizer=Adam --model="RILinkPredNet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=5000 --params_nnz=2 --depth=1 --use_sparse=True--name=RISparsemodel_linkpredDepth1WithFeatures
python model_linkpred.py --dataset=BlogCatalog --optimizer=Adam --model="RILinkPredNet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=5000 --params_nnz=2 --depth=2 --use_sparse=True--name=RISparsemodel_linkpredDepth2WithFeatures
python model_linkpred.py --dataset=BlogCatalog --optimizer=Adam --model="RILinkPredNet" --features_as="IndexVecs" --params_features_as="initialization_as_context" --params_dim=5000 --params_nnz=2 --depth=3 --use_sparse=True--name=RISparsemodel_linkpredDepth3WithFeatures


# Baseline
python3 model_linkpred.py --dataset=BlogCatalog --features_as=OneHotNodes --name=Baselinemodel_linkpredPubMedPublicWithoutFeatures --model=RILinkPredNet --depth=0;
python3 model_linkpred.py --dataset=BlogCatalog --features_as=OneHotNodes --name=Baselinemodel_linkpredPubMedPublicWithoutFeatures --model=RILinkPredNet --depth=1;
python3 model_linkpred.py --dataset=BlogCatalog --features_as=OneHotNodes --name=Baselinemodel_linkpredPubMedPublicWithoutFeatures --model=RILinkPredNet --depth=2;
python3 model_linkpred.py --dataset=BlogCatalog --features_as=OneHotNodes --name=Baselinemodel_linkpredPubMedPublicWithoutFeatures --model=RILinkPredNet --depth=3
# SparseNet Baseline
python3 model_linkpred.py --dataset=BlogCatalog --use_sparse=True --features_as=OneHotNodes --name=SparseBaselinemodel_linkpredPubMedPublicWithoutFeatures --model=RILinkPredNet --depth=0;
python3 model_linkpred.py --dataset=BlogCatalog --use_sparse=True --features_as=OneHotNodes --name=SparseBaselinemodel_linkpredPubMedPublicWithoutFeatures --model=RILinkPredNet --depth=1;
python3 model_linkpred.py --dataset=BlogCatalog --use_sparse=True --features_as=OneHotNodes --name=SparseBaselinemodel_linkpredPubMedPublicWithoutFeatures --model=RILinkPredNet --depth=2;
python3 model_linkpred.py --dataset=BlogCatalog --use_sparse=True --features_as=OneHotNodes --name=SparseBaselinemodel_linkpredPubMedPublicWithoutFeatures --model=RILinkPredNet --depth=3

# Without node features, dense model_linkpred
python3 model_linkpred.py --dataset=BlogCatalog --features_as=IndexVecs --model=RILinkPredNet --params_features_as=excluded --params_dim=4000 --params_nnz=2 --depth=0 --use_sparse=False --name=RISparsemodel_linkpredDepth0WithoutFeatures;
python3 model_linkpred.py --dataset=BlogCatalog --features_as=IndexVecs --model=RILinkPredNet --params_features_as=excluded --params_dim=4000 --params_nnz=2 --depth=1 --use_sparse=False --name=RISparsemodel_linkpredDepth1WithoutFeatures;
python3 model_linkpred.py --dataset=BlogCatalog --features_as=IndexVecs --model=RILinkPredNet --params_features_as=excluded --params_dim=4000 --params_nnz=2 --depth=2 --use_sparse=False --name=RISparsemodel_linkpredDepth2WithoutFeatures;
python3 model_linkpred.py --dataset=BlogCatalog --features_as=IndexVecs --model=RILinkPredNet --params_features_as=excluded --params_dim=4000 --params_nnz=2 --depth=3 --use_sparse=False --name=RISparsemodel_linkpredDepth3WithoutFeatures;

python3 model_linkpred.py --dataset=BlogCatalog --features_as=IndexVecs --model=RILinkPredNet --params_features_as=excluded --params_dim=4000 --params_nnz=4 --use_sparse=True --depth=0 --name=RIDensemodel_linkpredDepth0WithoutFeatures;
python3 model_linkpred.py --dataset=BlogCatalog --features_as=IndexVecs --model=RILinkPredNet --params_features_as=excluded --params_dim=4000 --params_nnz=4 --use_sparse=True --depth=1 --name=RIDensemodel_linkpredDepth1WithoutFeatures;
python3 model_linkpred.py --dataset=BlogCatalog --features_as=IndexVecs --model=RILinkPredNet --params_features_as=excluded --params_dim=4000 --params_nnz=4 --use_sparse=True --depth=2 --name=RIDensemodel_linkpredDepth2WithoutFeatures;
python3 model_linkpred.py --dataset=BlogCatalog --features_as=IndexVecs --model=RILinkPredNet --params_features_as=excluded --params_dim=4000 --params_nnz=4 --use_sparse=True --depth=3 --name=RIDensemodel_linkpredDepth3WithoutFeatures


python3 model_linkpred.py --dataset=BlogCatalog --features_as=IndexVecs --model=RILinkPredNet --params_features_as=excluded --params_dim=1500 --params_nnz=10 --depth=0 --use_sparse=True --name=RISparsemodel_linkpredDepth0WithoutFeatures;
python3 model_linkpred.py --dataset=BlogCatalog --features_as=IndexVecs --model=RILinkPredNet --params_features_as=excluded --params_dim=1500 --params_nnz=10 --depth=1 --use_sparse=True --name=RISparsemodel_linkpredDepth1WithoutFeatures;
python3 model_linkpred.py --dataset=BlogCatalog --features_as=IndexVecs --model=RILinkPredNet --params_features_as=excluded --params_dim=1500 --params_nnz=10 --depth=2 --use_sparse=True --name=RISparsemodel_linkpredDepth2WithoutFeatures;
python3 model_linkpred.py --dataset=BlogCatalog --features_as=IndexVecs --model=RILinkPredNet --params_features_as=excluded --params_dim=1500 --params_nnz=10 --depth=3 --use_sparse=True --name=RISparsemodel_linkpredDepth3WithoutFeatures
