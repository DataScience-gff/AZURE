library(usethis)
# Configurar el nombre de usuario en caso de no tenerlo 
usethis::use_git_config(user.name = "Jisaula", user.email = "juan.isaula@ficohsa.com")
use_git()
use_github()

# Generando Token en caso de no Tener 
usethis::create_github_token()

# Pegamos el token generado 
gitcreds::gitcreds_set()


# Pages
use_github_pages()


library(DBI)

con <- dbConnect(odbc::odbc(),
                 dsn = "SQLPools")
