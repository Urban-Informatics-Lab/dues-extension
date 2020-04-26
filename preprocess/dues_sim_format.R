library(tidyverse)
library(readxl)

# SMUD CSV File Paths

assessor_data <- 'D:/SMUD/XLSX/Service Address, Assessor\'s data, and EE program   Participation.xlsx'
energy_sim <- 'D:/SMUD/Simulation_Data/sim_full.csv'
ee_prog_ind <- 'D:/SMUD/EE_Program_indicators.csv'
service_addr <- 'D:/SMUD/Service_Addresses.csv'

# Project File Paths

building_energy_rds <- here::here("data/building_energy_sim.rds")

set.seed(5)

buildings <-
  read_xlsx(assessor_data, sheet = "Assessor\'s data") %>% 
  transmute(
    apn = APN,
    id = row_number(),
    year_built = as.integer(`ns1:YEAR_BUILT`),
    num_stories = as.integer(`ns1:NUMBER_OF_STORIES`),
    ground_floor = as.integer(`ns1:GROUND_FLOOR_GROSS`),
    net_rental = as.integer(`ns1:NET_RENTAL`),
    owner = as.character(`ns1:OWNER`),
    street_number = as.character(`ns1:STREET_NUMBER`),
    street_name = as.character(`ns1:STREET_NAME`),
    street_suffix = as.character(`ns1:STREET_SUFFIX`)
  ) %>%
  unite(col = "street", starts_with("street_"), sep = " ")

energy <- 
  read_csv(energy_sim) %>% 
  pivot_longer(
    cols = starts_with('Building '),
    names_to = "apn",
    names_pattern = "Building ?(.*) .*",
    values_to = "kwh"
  ) %>% 
  transmute(
    apn = str_c("00", apn),
    date_time = parse_date_time(`Date/Time`, "%m/%d/%Y %H:%M"),
    year = year(date_time),
    month = month(date_time),
    day = day(date_time),
    hour = hour(date_time),
    kwh = as.integer(kwh),
    freq = NA_integer_
  )

building_energy <- 
  buildings %>%
  left_join(energy, by = "apn") %>%
  select(id, apn, date_time, kwh, everything()) %>%
  write_rds(building_energy_rds)
