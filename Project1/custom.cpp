double computeCustomWeight(int docID,
		     int termID, 
		     int docTermFreq, 
		     double qryTermWeight,
		     Index *ind)
{
  return pow(qryTermWeight,2)*docTermFreq/ind->docCount(termID);
}

// compute the adjusted score
double computeCustomAdjustedScore(double origScore, // the score from the accumulator
			    int docID, // doc ID
			    Index *ind) // index
{
  return origScore/ind->docLength(docID);
}
