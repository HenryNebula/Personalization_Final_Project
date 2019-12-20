###Shiny App 
#
#load libraries
library(shiny)
library(leaflet)
library(tidyverse)
library(leaflet.extras)
library(rgdal)
library(ggplot2)
library(plotly)

dat<-read_csv("subset.csv")

cat_ls<-unique(dat$top_category)
user_ls<-c("All",unique(dat$user_id))

ui <- fluidPage(
    tabsetPanel(
        tabPanel("Top Reviewers",
                 sidebarLayout(
                     mainPanel( 
                         #this will create a space for us to display our map
                         tags$style(type = "text/css", "#mymap {height: calc(100vh - 20px) !important;}"),
                         leafletOutput(outputId = "mymap")
                     ),
                     
                     sidebarPanel(top = 60, left = 20, 
                                  selectizeInput("user",label="Select Users",#selected=user_ls[2],
                                                 choices=user_ls,multiple=TRUE,width="100%"),
                                  checkboxGroupInput("top_cat","Category",
                                                     cat_ls,
                                                     selected = cat_ls)
                                  
                     )
                 )
            ),
        tabPanel("Review Distribution",
                 htmlOutput("inc")
            
        )
    )
   
    )


server <- function(input, output, session) {
    pal2 <- colorFactor(
        palette = "viridis",
        domain = dat$user_id
    )
    
    output$leaf=renderUI({
        leafletOutput('mymap')
    })
    
    dataFiltered<-reactive({
        sel_cat<-input$top_cat
        sel_user<-input$user
        if("All"%in%sel_user){sel_user<-user_ls[-1]}
        dat%>%filter(top_category%in%sel_cat,user_id%in%sel_user)
    })
    
    #create the map
    output$mymap <- renderLeaflet({
        lf<-leaflet() %>% 
            setView(-98.58, 39.82,  zoom = 4)%>%addTiles()
    
        if(nrow(dataFiltered())>0){
            lf<-lf%>%
                addCircles(data = dataFiltered(), 
                       lat = ~ latitude, lng = ~ longitude, 
                       weight = 1, 
                       radius = 50, 
                       label = ~as.character(user_id),
                       color = ~pal2(user_id), fillOpacity = 0.5)}
        
        lf
    })
    
    
    getPage<-function() {
        return(includeHTML("mymap.html"))
    }
    output$inc<-renderUI({getPage()})
    
}

shinyApp(ui, server)

