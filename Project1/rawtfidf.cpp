// compute the weight of a matched term
double computeRawTFIDFWeight(int docID,
		     int termID, 
		     int docTermFreq, 
		     double qryTermWeight,
		     Index *ind)
{
  return docTermFreq*log2(ind->docCount()/ind->docCount(termID))*qryTermWeight;
}


