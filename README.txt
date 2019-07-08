This repo contains codes to jointly summarize tweets using an ILP based unsupervised approach based on path construction.

###########################################################################################################################################

Step 1: Extract concepts from raw AIDR classified tweets
python tweet_concept_extraction.py /demo/infrastructure_20150425.txt place/nepal_place.txt infrastructure_concept_20150425.txt

Step 2: Generate 1000 word summary using COWTS [Rudra et al. CIKM 2015]
python NCOWTS.py /demo/infrastructure_concept_20150425.txt place/nepal_place.txt infrastructure 20150425 1000

Step 3: Tag above generated 1000 word summary
python tag_top_1000.py demo/infrastructure_ICOWTS_20150425.txt demo/infrastructure_icowts_tagged_20150425.txt

Step 4: Generate Path from the tagged data
	A. Go to TwitterSumm/absummarizer
	B. In bigram_path_generation.py set the base path
python bigram_path_generation.py infrastructure_icowts_tagged_20150425.txt

	C. It will generate two files: 1. infrastructure_icowts_tagged_20150425.txt_paths 2. infrastructure_icowts_tagged_20150425.txt_details.txt

Step 5: Generate final summary
python abstractive_summary.py demo/infrastructure_icowts_tagged_20150425.txt_paths infrastructure place/nepal_place.txt 20150425 200

################################################################################################################################################

NOTE:
	A. Set the paths accordingly
	B. In all the codes set the Twitter Tagger path accordingly
	C. Demo directory contains some sample files just to demonstrate format of files passing from one module to the next one. To generate summaries kindly apply the methods over original dataset.
	D. For the path generation module [TwitterSumm], kindly drop a mail.

Queries: koustav.rudra@cse.iitkgp.ernet.in  [Path generation module was originally developed by Siddhartha Banerjee. If you have any queries regarding path generation step kindly forward queries to sbanerjee@ist.psu.edu]
