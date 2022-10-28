python gcn.py --dataset=Cora --split=complete --optimizer=Adam --use_sparse=True --model=RINet --depth=0
python gcn.py --dataset=Cora --split=complete --optimizer=Adam --use_sparse=True --model=RINet --depth=1
python gcn.py --dataset=Cora --split=complete --optimizer=Adam --use_sparse=True --model=RINet --depth=2
python gcn.py --dataset=Cora --split=complete --optimizer=Adam --use_sparse=True --model=RINet --depth=3
python gcn.py --dataset=Cora --split=complete --optimizer=Adam --use_sparse=True --model=RINet --depth=0 --features_as=OneHotNodes
python gcn.py --dataset=Cora --split=complete --optimizer=Adam --use_sparse=True --model=RINet --depth=1 --features_as=OneHotNodes
python gcn.py --dataset=Cora --split=complete --optimizer=Adam --use_sparse=True --model=RINet --depth=2 --features_as=OneHotNodes
python gcn.py --dataset=Cora --split=complete --optimizer=Adam --use_sparse=True --model=RINet --depth=3 --features_as=OneHotNodes

python gcn.py --dataset=CiteSeer --split=complete --optimizer=Adam --use_sparse=True --model=RINet --depth=0
python gcn.py --dataset=CiteSeer --split=complete --optimizer=Adam --use_sparse=True --model=RINet --depth=1
python gcn.py --dataset=CiteSeer --split=complete --optimizer=Adam --use_sparse=True --model=RINet --depth=2
python gcn.py --dataset=CiteSeer --split=complete --optimizer=Adam --use_sparse=True --model=RINet --depth=3

python gcn.py --dataset=CiteSeer --split=complete --optimizer=Adam --use_sparse=True --model=RINet --depth=0 --features_as=OneHotNodes
python gcn.py --dataset=CiteSeer --split=complete --optimizer=Adam --use_sparse=True --model=RINet --depth=1 --features_as=OneHotNodes
python gcn.py --dataset=CiteSeer --split=complete --optimizer=Adam --use_sparse=True --model=RINet --depth=2 --features_as=OneHotNodes
python gcn.py --dataset=CiteSeer --split=complete --optimizer=Adam --use_sparse=True --model=RINet --depth=3 --features_as=OneHotNodes



python gcn.py --dataset=PubMed --split=complete --optimizer=Adam --use_sparse=True --model=RINet --depth=0
python gcn.py --dataset=PubMed --split=complete --optimizer=Adam --use_sparse=True --model=RINet --depth=2
python gcn.py --dataset=PubMed --split=complete --optimizer=Adam --use_sparse=True --model=RINet --depth=1
python gcn.py --dataset=PubMed --split=complete --optimizer=Adam --use_sparse=True --model=RINet --depth=3

python gcn.py --dataset=PubMed --split=complete --optimizer=Adam --use_sparse=True --model=RINet --depth=0 --features_as=OneHotNodes
python gcn.py --dataset=PubMed --split=complete --optimizer=Adam --use_sparse=True --model=RINet --depth=2 --features_as=OneHotNodes
python gcn.py --dataset=PubMed --split=complete --optimizer=Adam --use_sparse=True --model=RINet --depth=1 --features_as=OneHotNodes
python gcn.py --dataset=PubMed --split=complete --optimizer=Adam --use_sparse=True --model=RINet --depth=3 --features_as=OneHotNodes



python model_linkpred.py --dataset=Cora --split=complete --optimizer=Adam --use_sparse=True --model=RILinkPredNet --depth=0
python model_linkpred.py --dataset=Cora --split=complete --optimizer=Adam --use_sparse=True --model=RILinkPredNet --depth=1
python model_linkpred.py --dataset=Cora --split=complete --optimizer=Adam --use_sparse=True --model=RILinkPredNet --depth=2
python model_linkpred.py --dataset=Cora --split=complete --optimizer=Adam --use_sparse=True --model=RILinkPredNet --depth=3

python model_linkpred.py --dataset=Cora --split=complete --optimizer=Adam --use_sparse=True --model=RILinkPredNet --depth=0 --features_as=OneHotNodes
python model_linkpred.py --dataset=Cora --split=complete --optimizer=Adam --use_sparse=True --model=RILinkPredNet --depth=1 --features_as=OneHotNodes
python model_linkpred.py --dataset=Cora --split=complete --optimizer=Adam --use_sparse=True --model=RILinkPredNet --depth=2 --features_as=OneHotNodes
python model_linkpred.py --dataset=Cora --split=complete --optimizer=Adam --use_sparse=True --model=RILinkPredNet --depth=3 --features_as=OneHotNodes



python model_linkpred.py --dataset=CiteSeer --split=complete --optimizer=Adam --use_sparse=True --model=RILinkPredNet --depth=0
python model_linkpred.py --dataset=CiteSeer --split=complete --optimizer=Adam --use_sparse=True --model=RILinkPredNet --depth=1
python model_linkpred.py --dataset=CiteSeer --split=complete --optimizer=Adam --use_sparse=True --model=RILinkPredNet --depth=2
python model_linkpred.py --dataset=CiteSeer --split=complete --optimizer=Adam --use_sparse=True --model=RILinkPredNet --depth=3

python model_linkpred.py --dataset=CiteSeer --split=complete --optimizer=Adam --use_sparse=True --model=RILinkPredNet --depth=0 --features_as=OneHotNodes
python model_linkpred.py --dataset=CiteSeer --split=complete --optimizer=Adam --use_sparse=True --model=RILinkPredNet --depth=1 --features_as=OneHotNodes
python model_linkpred.py --dataset=CiteSeer --split=complete --optimizer=Adam --use_sparse=True --model=RILinkPredNet --depth=2 --features_as=OneHotNodes
python model_linkpred.py --dataset=CiteSeer --split=complete --optimizer=Adam --use_sparse=True --model=RILinkPredNet --depth=3 --features_as=OneHotNodes



python model_linkpred.py --dataset=PubMed --split=complete --optimizer=Adam --use_sparse=True --model=RILinkPredNet --depth=0
python model_linkpred.py --dataset=PubMed --split=complete --optimizer=Adam --use_sparse=True --model=RILinkPredNet --depth=2
python model_linkpred.py --dataset=PubMed --split=complete --optimizer=Adam --use_sparse=True --model=RILinkPredNet --depth=1
python model_linkpred.py --dataset=PubMed --split=complete --optimizer=Adam --use_sparse=True --model=RILinkPredNet --depth=3

python model_linkpred.py --dataset=PubMed --split=complete --optimizer=Adam --use_sparse=True --model=RILinkPredNet --depth=0 --features_as=OneHotNodes
python model_linkpred.py --dataset=PubMed --split=complete --optimizer=Adam --use_sparse=True --model=RILinkPredNet --depth=2 --features_as=OneHotNodes
python model_linkpred.py --dataset=PubMed --split=complete --optimizer=Adam --use_sparse=True --model=RILinkPredNet --depth=1 --features_as=OneHotNodes
python model_linkpred.py --dataset=PubMed --split=complete --optimizer=Adam --use_sparse=True --model=RILinkPredNet --depth=3 --features_as=OneHotNodes
