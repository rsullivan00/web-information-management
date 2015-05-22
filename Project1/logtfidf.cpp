// compute the weight of a matched term
double computeLogTFIDFWeight(int docID,
		     int termID, 
		     int docTermFreq, 
		     double qryTermWeight,
		     Index *ind)
{
  return (log2(docTermFreq) + 1)*log2(ind->docCount()/ind->docCount(termID))*qryTermWeight;
}


