library(castor)
library(nloptr)
library(scales)

bisse_mle = function(x) {
    
    # defaults
    num_trials   = 10
    num_scouts   = 30
    num_states   = 2
    max_model_runtime = 5
    
    # get analysis args
    idx = x$idx
    phy_fn = x$phy_fn
    dat_fn = x$dat_fn
    sample_frac = x$sample_frac
    par_true = x$par_true
    
    # load input dataset
    phy = read.tree(phy_fn)
    dat_df = read.csv(dat_fn, header=T)
    dat = dat_df$data + 1 # convert to base-1
    names(dat) = dat_df$taxa
    dat = dat[phy$tip.label]
    
    # generate MLE
    par_mle = NULL
    while (is.null(par_mle)) {
        par_mle = tryCatch(
        {
            # first guess
            birth = par_true[1:2] * exp(runif(2, -1, 1))
            death = par_true[3] * exp(runif(2, -1, 1))
            state_rate = par_true[4] * exp(runif(1, -1, 1))
            Q = matrix(state_rate, ncol=2, nrow=2)
            diag(Q) = 0; diag(Q) = -rowSums(Q)
            
            first_guess = list(
                birth_rates = birth,
                death_rates = death,
                transition_matrix = Q
            )
            
            # bounds
            lower=list(
                birth_rates=0.0,
                death_rates=0.0,
                transition_matrix=0.0
            )
            
            upper=list(
                birth_rates=1,
                death_rates=1,
                transition_matrix=1
            )
            
            # get MLE
            ret=fit_musse(
                tree=phy,
                tip_pstates=dat,
                birth_rate_model="ARD",
                death_rate_model="ER",
                transition_rate_model="ER",
                sampling_fractions=sample_frac,
                first_guess=first_guess,
                #root_conditioning="crown",
                lower=lower,
                upper=upper,
                #optim_algorithm="nloptr",
                Nstates=num_states,
                Ntrials=num_trials,
                Nscouts=num_scouts,
                max_model_runtime=max_model_runtime,
                verbose=F,
                diagnostics=F)
            
            #print(ret$parameters)
            ret$parameters
        },
        error = function(m) {
            NULL
        })
        
        # save MLEs
        out_mle = c( par_mle$birth_rates, par_mle$death_rates[1], par_mle$transition_matrix[1,2])
    }
    x$par_mle = log( out_mle, base=10 )
    names(x$par_mle) = names(x$par_cnn_point) = names(x$par_cnn_lower) = names(x$par_cnn_upper) = names(x$par_true) = param_names
    return(x)
}


do_compare = function(x, y, par_names=NA) {
    
    mae = colMeans(abs(x-y))
    mse = colMeans((x-y)^2)
    rmse = sqrt(mse)
    mape = colMeans(abs(x-y)/abs(y))
    
    
    slope=c()
    intercept=c()
    r2=c()
    for (i in 1:ncol(x)) {
        dat = data.frame(x=x[,i], y=y[,i])
        m = lm(y~x, data=dat)
        intercept[i] = m$coefficients[1]
        slope[i] = m$coefficients[2]
        r2[i] = summary(m)$r.squared
    }
    names(slope)=names(intercept)=names(r2) = colnames(x)
    
    ret = data.frame(mae=mae, mse=mse, rmse=rmse, mape=mape, slope=slope, intercept=intercept, r2=r2)

    return(ret)
}

make_compare_table = function(x) {
    
    par_true = x$par_true
    par_cnn_point  = x$par_cnn_point
    par_mle  = x$par_mle

    res_cnn_true = do_compare(par_cnn_point, par_true)
    res_mle_true = do_compare(par_mle, par_true)
    res_cnn_mle = do_compare(par_cnn_point, par_mle)

    ret=list(
        cnn_true=res_cnn_true,
        mle_true=res_mle_true,
        cnn_mle=res_cnn_mle
    )
        
    return(ret)
}


make_input_table = function(x, unlog=T) {

    
    # create result matrices
    par_true       = matrix( unlist(lapply(x, function(y) { y$par_true })), ncol=4, byrow=T )
    par_cnn_point  = matrix( unlist(lapply(x, function(y) { y$par_cnn_point })), ncol=4, byrow=T )
    par_cnn_lower  = matrix( unlist(lapply(x, function(y) { y$par_cnn_lower })), ncol=4, byrow=T )
    par_cnn_upper  = matrix( unlist(lapply(x, function(y) { y$par_cnn_upper })), ncol=4, byrow=T )
    par_mle        = matrix( unlist(lapply(x, function(y) { y$par_mle })), ncol=4, byrow=T )

    if (unlog) {
        par_true      = 10^par_true
        par_cnn_point = 10^par_cnn_point
        par_cnn_lower = 10^par_cnn_lower
        par_cnn_upper = 10^par_cnn_upper
        par_mle       = 10^par_mle
    }
    
    # assign parameter names
    par_names = x[[1]]$param_names
    colnames(par_true)=colnames(par_cnn_point)=colnames(par_cnn_lower)=colnames(par_cnn_upper)=colnames(par_mle)=par_names
    
    # return results
    ret=list(
        par_true=par_true,
        par_mle=par_mle,
        par_cnn_point=par_cnn_point,
        par_cnn_lower=par_cnn_lower,
        par_cnn_upper=par_cnn_upper
    )
    
    return(ret)
}


plot_comparison = function(dat, stat, out_fn="validate_bisse.pdf") {

    param_name = colnames(dat$par_true)
    num_param = length(param_name)
    
    pdf(out_fn, width=10, height=7)
    
     #mai=c(0.5, 0.6, 0.1, 0.1), 
    par(
        mfrow=c(3,num_param),
        mar=c(4.5, 4.5, 1.5, 1.5)
    )
    
    vmax = c()
    for (i in 1:num_param) {
        vmax[i] = max(rbind(dat$par_true, dat$par_cnn, dat$par_mle)[,i] )
    }
    vmin = c()
    for (i in 1:num_param) {
        vmin[i] = min(rbind(dat$par_true, dat$par_cnn, dat$par_mle)[,i] )
    }
    
    alpha = 0.5
    alpha_blue = alpha("blue", alpha)
    alpha_gray = alpha("darkgray", alpha)
    alpha_red = alpha("red", alpha)
    alpha_gold = alpha("gold2", alpha)
    
    for (i in 1:num_param) {
        #is_covered = (dat$par_cnn_lower[,i] <= dat$par_true[,i] & dat$par_cnn_upper[,i] >= dat$par_true[,i])
        # covg = round(sum(is_covered) / length(is_covered), digits = 2) * 100
        is_covered = rep(T, nrow(dat$par_true))
        plot(dat$par_true[is_covered,i], dat$par_cnn_point[is_covered,i], xlab="true", ylab="CNN",
            col=alpha_blue, pch=16, alpha=0.5, main=param_names[i],
            xlim=c(vmin[i],vmax[i]), ylim=c(vmin[i],vmax[i]))
        points(dat$par_true[!is_covered,i], dat$par_cnn_point[!is_covered,i],
               col=alpha_gray, pch=16, alpha=0.5)
        # segments(x0=dat$par_true[is_covered,i], x1=dat$par_true[is_covered,i],
        #          y0=dat$par_cnn_lower[is_covered,i], y1=dat$par_cnn_upper[is_covered,i],
        #          col=alpha_blue, alpha=0.5, lwd=0.5)
        # segments(x0=dat$par_true[!is_covered,i], x1=dat$par_true[!is_covered,i],
        #          y0=dat$par_cnn_lower[!is_covered,i], y1=dat$par_cnn_upper[!is_covered,i],
        #          col=alpha_gray, alpha=0.5, lwd=0.5)
        text(x=vmin[i], y=vmax[i]*1.0, adj=c(0.0, 1.0), label=paste("R^2:", round(stat$cnn_true$r2[i], digits=2)))
        # text(x=vmin[i], y=vmax[i]*0.95, adj=c(0.0, 1.0), label=paste("covg:", round(sum(is_covered) / length(is_covered), digits=2)))
        abline(a=0,b=1,col="black",lwd=1)
        abline(a=stat$cnn_true$intercept[i], b=stat$cnn_true$slope[i], col="blue",lwd=1, lty=1)
    }
    
    for (i in 1:num_param) {
        plot(dat$par_true[,i], dat$par_mle[,i], xlab="true", ylab="MLE",
            col=alpha_red, pch=16, alpha=0.5, main=param_names[i],
            xlim=c(vmin[i],vmax[i]), ylim=c(vmin[i],vmax[i]))
        text(x=vmin[i], y=vmax[i], adj=c(0,1), label=paste("R^2:",round(stat$mle_true$r2[i], digits=2)))
        abline(a=0,b=1,col="black",lwd=1)
        abline(a=stat$mle_true$intercept[i], b=stat$mle_true$slope[i], col="red",lwd=1, lty=1)
    }
    
    for (i in 1:num_param) {
        plot(dat$par_cnn_point[,i], dat$par_mle[,i], xlab="CNN", ylab="MLE",
            col=alpha_gold, pch=16, alpha=0.5, main=param_names[i],
            xlim=c(vmin[i],vmax[i]), ylim=c(vmin[i],vmax[i]))
        text(x=vmin[i], y=vmax[i], adj=c(0,1), label=paste("R^2:",round(stat$cnn_mle$r2[i], digits=2)))
        abline(a=0,b=1,col="black",lwd=1)
        abline(a=stat$cnn_mle$intercept[i], b=stat$cnn_mle$slope[i], col="gold2",lwd=1, lty=1)
    }
    
    # done!
    dev.off()
}

save_tables = function(x) {
    write.csv(x=x$par_true, file="./bisse_par_true.csv", sep=",", quote=F, row.names=F)
    write.csv(x=x$par_mle, file="./bisse_par_mle.csv", sep=",", quote=F, row.names=F)
    write.csv(x=x$par_cnn_point, file="./bisse_par_cnn.csv", sep=",", quote=F, row.names=F)
    return
}