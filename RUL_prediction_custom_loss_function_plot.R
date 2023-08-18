x <- seq(-60, 60)
y <- function(d){
	if (d<0){
		value <- exp(-d/10)-1	
		#value <- -5*d
	}else{
		value <- exp(d/13)-1
	}
}

y2 <- function(d){
	value <- d^2
}

plot(x, sapply(x,y), ylim=c(0, 500))