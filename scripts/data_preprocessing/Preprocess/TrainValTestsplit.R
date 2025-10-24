TrainValTestsplit <- function(dataframe, trainindex, valindex, testindex){
  if(ncol(dataframe)>1){
    return(list(train=dataframe[trainindex,], val=dataframe[valindex,],test=dataframe[testindex,]))
  }else{
    frame1 <- data.frame(dataframe[trainindex,])
    frame2 <- data.frame(dataframe[valindex,])
    frame3 <- data.frame(dataframe[testindex,])
    names(frame1) <- names(dataframe)
    names(frame2) <- names(dataframe)
    names(frame3) <- names(dataframe)
    return(list(train=frame1, val=frame2,test=frame3))
  }
}