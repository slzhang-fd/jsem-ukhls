.onLoad <- function(libname, pkgname) {
  setjsem_threads(3)
}

.onAttach <- function(libname, pkgname){
  v = packageVersion("jsem")
  packageStartupMessage("jsem ", v, " using ", getjsem_threads(), " threads (see ?getjsem_threads())")
}