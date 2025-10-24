equalSizedBins <- function(phenoAvg, trainindex, valindex, testindex) {
  phenoVal = phenoAvg[valindex]
  phenoTest = phenoAvg[testindex]
  phenoAvg = phenoAvg[trainindex]

	## equal sized bins 
        q = quantile(phenoAvg, probs=c(1/3,2/3), na.rm=TRUE)

	minX = min(phenoAvg, na.rm=TRUE)
	maxX = max(phenoAvg, na.rm=TRUE)

	phenoBinned = phenoAvg;
	if (q[1]==minX) {
		# edge case - quantile value is lowest value

		# assign min value as cat1
		idx1 = which(phenoAvg==q[1]);
		phenoBinned[idx1] = 0;

		# divide remaining values into cat2 and cat3
		phenoAvgRemaining = phenoAvg[which(phenoAvg!=q[1])];
		qx = quantile(phenoAvgRemaining, probs=c(0.5), na.rm=TRUE)
		minXX = min(phenoAvgRemaining, na.rm=TRUE)
		maxXX =	max(phenoAvgRemaining, na.rm=TRUE)

		if (qx[1]==minXX) {
			# edge case again - quantile value is lowest value
			idx2 = which(phenoAvg==qx[1]);
			idx3 = which(phenoAvg>qx[1]);
			validx1 = which(phenoVal<=q[1])
			validx2 = which(phenoVal<=qx[1] & phenoVal>q[1])
			validx3 = which(phenoVal>qx[1])
			
			testidx1 = which(phenoTest<=q[1])
			testidx2 = which(phenoTest<=qx[1] & phenoTest>q[1])
			testidx3 = which(phenoTest>qx[1])
			cat("Bin 1: <=", q[1],", bin 2: > ",q[1], " AND <=", qx[1] , "bin 3: >", qx[1], " || ", sep="")
		}
		else if (qx[1]==maxXX) {
			# edge case again - quantile value is max value
			idx2 = which(phenoAvg<qx[1] & phenoAvg>q[1]);
			idx3 = which(phenoAvg==qx[1]);
			validx1 = which(phenoVal<=q[1])
			validx2 = which(phenoVal<qx[1] & phenoVal>q[1])
			validx3 = which(phenoVal>=qx[1])
			
			testidx1 = which(phenoTest<=q[1])
			testidx2 = which(phenoTest<qx[1] & phenoTest>q[1])
			testidx3 = which(phenoTest>=qx[1])
			cat("Bin 1: <=", q[1],", bin 2: > ",q[1], " AND <", qx[1] , "bin 3: >=", qx[1], " || ", sep="")
		}
		else {
			idx2 = which(phenoAvg<qx[1] & phenoAvg>q[1]);
			idx3 = which(phenoAvg>=qx[1]);
			validx1 = which(phenoVal<=q[1])
			validx2 = which(phenoVal<qx[1] & phenoVal>q[1])
			validx3 = which(phenoVal>=qx[1])
			
			testidx1 = which(phenoTest<=q[1])
			testidx2 = which(phenoTest<qx[1] & phenoTest>q[1])
			testidx3 = which(phenoTest>=qx[1])
			cat("Bin 1: ==", q[1],", bin 2: > ",q[1], " AND <", qx[1] , "bin 3: >=", qx[1], " || ", sep="")
		}
		phenoBinned[idx2] = 1;
    phenoBinned[idx3] = 2;
		phenoVal[validx1] = 0
		phenoVal[validx2] = 1
		phenoVal[validx3] = 2
		phenoTest[testidx1] = 0
		phenoTest[testidx2] = 1
		phenoTest[testidx3] = 2
	}
	else if (q[2]==maxX) {
		# edge case - quantile value is highest value

		# assign max value as cat3
		idx3 = which(phenoAvg==q[2]);
		phenoBinned[idx3] = 2;

		# divide remaining values into cat1 and cat2
		phenoAvgRemaining = phenoAvg[which(phenoAvg!=q[2])];
                qx = quantile(phenoAvgRemaining, probs=c(0.5), na.rm=TRUE)
		minXX = min(phenoAvgRemaining, na.rm=TRUE)
                maxXX = max(phenoAvgRemaining, na.rm=TRUE)

		if (qx[1]==minXX) {
			# edge case again - quantile value is lowest value
                        idx1 = which(phenoAvg==qx[1]);
                        idx2 = which(phenoAvg>qx[1] & phenoAvg<q[2]);
                        validx1 = which(phenoVal <= qx[1])
                        validx2 = which(phenoVal > qx[1] & phenoVal < q[2])
                        validx3 = which(phenoVal >= q[2])
                        testidx1 = which(phenoTest <= qx[1])
                        testidx2 = which(phenoTest > qx[1] & phenoTest < q[2])
                        testidx3 = which(phenoTest >= q[2])
			cat("Bin 1: <=", qx[1], ", bin 2: >", qx[1], " AND < ", q[2], ", bin 3: >=", q[2], " || ", sep="")
                }
                else if	(qx[1]==maxXX) {
			# edge case again - quantile value is max value
                        idx1 = which(phenoAvg<qx[1]);
                        idx2 = which(phenoAvg==qx[1]);
                        validx1 = which(phenoVal < qx[1])
                        validx2 = which(phenoVal >= qx[1] & phenoVal < q[2])
                        validx3 = which(phenoVal >= q[2])
                        testidx1 = which(phenoTest < qx[1])
                        testidx2 = which(phenoTest >= qx[1] & phenoTest < q[2])
                        testidx3 = which(phenoTest >= q[2])
			cat("Bin 1: <", qx[1], ", bin 2: >=", qx[1], " AND < ", q[2], ", bin 3: >=", q[2], " || ", sep="")
                }
                else {
	                idx1 = which(phenoAvg<qx[1]);  
			idx2 = which(phenoAvg>=qx[1] & phenoAvg<q[2]);
			validx1 = which(phenoVal < qx[1])
			validx2 = which(phenoVal >= qx[1] & phenoVal < q[2])
			validx3 = which(phenoVal >= q[2])
			testidx1 = which(phenoTest < qx[1])
			testidx2 = which(phenoTest >= qx[1] & phenoTest < q[2])
			testidx3 = which(phenoTest >= q[2])
			cat("Bin 1: <", qx[1], ", bin 2: >=", qx[1], " AND < ", q[2], ", bin 3: >=", q[2], " || ", sep="")
		}

                phenoBinned[idx1] = 0;
                phenoBinned[idx2] = 1;
                phenoVal[validx1] = 0
                phenoVal[validx2] = 1
                phenoVal[validx3] = 2
                phenoTest[testidx1] = 0
                phenoTest[testidx2] = 1
                phenoTest[testidx3] = 2
	}
       else if (q[1] == q[2]) {
		# both quantiles correspond to the same value so set 
		# cat1 as < this value, cat2 as exactly this value and
		# cat3 as > this value
              	phenoBinned = phenoAvg;
        	idx1 = which(phenoAvg<q[1]);
                idx2 = which(phenoAvg==q[2]);
                idx3 = which(phenoAvg>q[2]);
                validx1 = which(phenoVal<q[1])
                validx2 = which(phenoVal==q[2])
                validx3 = which(phenoVal>q[2])
                testidx1 = which(phenoTest<q[1]);
                testidx2 = which(phenoTest==q[2]);
                testidx3 = which(phenoTest>q[2]);
                phenoBinned[idx1] = 0;
                phenoBinned[idx2] = 1;
                phenoBinned[idx3] = 2;
                phenoVal[validx1] = 0
                phenoVal[validx2] = 1
                phenoVal[validx3] = 2
                phenoTest[testidx1] = 0
                phenoTest[testidx2] = 1
                phenoTest[testidx3] = 2

		cat("Bin 1: <", q[1], ", bin 2: ==", q[2], ", bin 3: >", q[2], " || ", sep="")
       	}
        else {
		# standard case - split the data into three roughly equal parts where
		# cat1<q1, cat2 between q1 and q2, and cat3>=q2
         	phenoBinned = phenoAvg;
                idx1 = which(phenoAvg<q[1]);
                idx2 = which(phenoAvg>=q[1] & phenoAvg<q[2]);
               	idx3 = which(phenoAvg>=q[2]);
               	validx1 = which(phenoVal<q[1])
               	validx2 = which(phenoVal>=q[1] & phenoVal<q[2])
               	validx3 = which(phenoVal>=q[2])
               	testidx1 = which(phenoTest<q[1])
               	testidx2 = which(phenoTest>=q[1] & phenoTest<q[2])
               	testidx3 = which(phenoTest>=q[2])
                phenoBinned[idx1] = 0;
                phenoBinned[idx2] = 1;
                phenoBinned[idx3] = 2;
                phenoVal[validx1] = 0
                phenoVal[validx2] = 1
                phenoVal[validx3] = 2
                phenoTest[testidx1] = 0
                phenoTest[testidx2] = 1
                phenoTest[testidx3] = 2

		cat("Bin 1: <", q[1], ", bin 2: >=", q[1], "AND < ", q[2] ,", bin 3: >=", q[2], " || ", sep="")
	}

	cat("cat N: ", length(idx1),", ",length(idx2),", ",length(idx3), " || ", sep="");
	
	return(list(train=phenoBinned, val=phenoVal, test=phenoTest));
}

