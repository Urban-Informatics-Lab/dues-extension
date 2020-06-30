library(tidyverse)

# SMUD CSV File Paths

assessor_data <- 'D:/SMUD/Assessor_Data.csv'
energy_ami <- 'D:/SMUD/Energy_AMI.csv'
ee_prog_ind <- 'D:/SMUD/EE_Program_indicators.csv'
service_addr <- 'D:/SMUD/Service_Addresses.csv'

# Project File Paths

building_energy_rds <- here::here("data/building_energy_actual.rds")

set.seed(5)

buildings <-
  read_csv(assessor_data) %>%
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
  read_csv(energy_ami) %>%
  pivot_longer(
    cols = starts_with('kwh_'),
    names_to = "hour",
    names_pattern = "kwh_?(.*)",
    values_to = "kwh"
  ) %>%
  transmute(
    apn = APN,
    date_time = str_c(Date, "/", hour) %>% parse_date_time("%m/%d/%Y/%H"),
    year = year(date_time),
    month = month(date_time),
    day = day(date_time),
    hour = hour(date_time),
    freq = as.integer(`_FREQ_`),
    kwh,
    # kwh = as.integer(kwh),
  )

building_energy <-
  buildings %>%
  left_join(energy, by = "apn") %>%
  select(id, apn, date_time, kwh, everything()) %>%
  write_rds(building_energy_rds)
