library(tidyverse)
library(data.table)
library(ggsci)
library(ggpubr)
library(scales)
library(forcats)


# load data ---------------------------------------------------------------

df <- fread("../../results/Preprocess/phenofile_origin.csv") %>% as_tibble()
showcase <- read_csv("../../data/Preprocess/showcase.csv")

fields_baseline <- showcase %>%
  filter(Class_top == "Baseline characteristics") %>%
  pull(FieldID) %>%
  as.character()

fields_lifestyle <- showcase %>%
  filter(Class_top == "Life") %>%
  pull(FieldID) %>%
  as.character()

fields_measurement <- showcase %>%
  filter(Class_top == "Measures") %>%
  pull(FieldID) %>%
  as.character()

fields_environment <- showcase %>%
  filter(Class_top == "Natural and social environment") %>%
  pull(FieldID) %>%
  as.character()

fields_genetic <- showcase %>%
  filter(Class_top == "Genetic") %>%
  pull(FieldID) %>%
  as.character()

names <- sapply(strsplit(colnames(df), "[#, -]"), function(ls) ls[1])
inds_baseline <- which(names %in% fields_baseline)
inds_lifestyle <- which(names %in% fields_lifestyle)
inds_measurement <- which(names %in% fields_measurement)
inds_environment <- which(names %in% fields_environment)
inds_genetic <- which(names %in% fields_genetic)

setdiff(names, Reduce(union, list(fields_baseline, fields_lifestyle, fields_measurement, fields_environment, fields_genetic)))

df_baseline <- df[, c(1, inds_baseline)]
df_lifestyle <- df[, c(1, inds_lifestyle)]
df_measurement <- df[, c(1, inds_measurement)]
df_environment <- df[, c(1, inds_environment)]
df_genetic <- df[, c(1, inds_genetic)]
df_all <- df[, c(1, inds_baseline, inds_lifestyle, inds_measurement, inds_environment, inds_genetic)]

# helpful functions -------------------------------------------------------
missing_rate_histogram <- function(tbl, color_num = 1, ymax = 200, binwidth = 5) {
  
  missing_rates <- (sapply(tbl, function(x) sum(is.na(x)) / length(x)) * 100)[-1]
  missing_tbl <- tibble(
    column = names(missing_rates),
    missing_rate = missing_rates
  )
  
  ggplot(missing_tbl) +
    aes(x = missing_rate) +
    geom_histogram(fill = pal_npg("nrc")(10)[color_num], binwidth = binwidth, boundary = 0) +
    theme_bw() +
    labs(
      x = "Missing Rate (%)",
      y = "Number of Columns"
    ) +
    scale_y_continuous(limits = c(0, ymax)) +
    # scale_x_continuous(limits = c(0, 100)) +
    theme(text = element_text(size=12))
}
missing_rate_histogram_row <- function(tbl, color_num = 1, binwidth = 5) {
  
  missing_rates <- apply(tbl[,-1], 1, function(x) sum(is.na(x)) / length(x)) * 100
  missing_tbl <- tibble(
    column = names(missing_rates),
    missing_rate = missing_rates
  )
  
  ggplot(missing_tbl) +
    aes(x = missing_rate) +
    geom_histogram(fill = pal_npg("nrc")(10)[color_num], binwidth = binwidth, boundary = 0) +
    theme_bw() +
    labs(
      x = "Missing Rate (%)",
      y = "Number of Rows"
    ) +
    # scale_y_continuous(limits = c(0, ymax)) +
    scale_x_continuous(limits = c(0, 100)) +
    theme(text = element_text(size=12))
}
c25 <- c( "#E31A1C", "dodgerblue2", "green4",  "#6A3D9A", "#FF7F00",   "black", "gold1",  "skyblue2", "#FB9A99",  "palegreen2",  "#CAB2D6",  "#FDBF6F", 
          "gray70", "khaki2",  "maroon", "orchid1", "deeppink1", "blue1", "steelblue4",  "darkturquoise", "green1", "yellow4", "yellow3",  "darkorange4", "brown")
showcase <- read_csv("../../data/Preprocess/showcase.csv")

# prepare center_time ------------------------------------------------------------------
time_center <- read_csv("../../data/Visualize_missingness/time_center.csv") %>%
  rename(time = `53-0.0`, center = `54-0.0`) %>%
  mutate(center = as.factor(center))

var4598_time_center <- df_all %>%
  select(eid, `4598-0.0`) %>%
  inner_join(time_center, by = "eid")

coding10 <- read_tsv("../../data/Visualize_missingness/coding10.tsv") %>%
  mutate(center = as.factor(coding)) %>%
  select(-coding)

missing_rate_by_center <- var4598_time_center %>%
  group_by(center) %>%
  summarize(across(`4598-0.0`, ~mean(is.na(.)), .names = "missing_rate_4598")) %>%
  inner_join(coding10, by = "center") %>%
  mutate(meaning = factor(meaning, levels = (.$meaning))) %>%
  select(missing_rate_4598, meaning) %>%
  rename(center = "meaning")

center_meaning <- time_center %>%
  group_by(center) %>%
  summarise(count = n()) %>%
  inner_join(coding10) %>%
  mutate(meaning = factor(meaning, levels = .$meaning)) 

df_pilot <- df_all %>%
  inner_join(
    time_center %>%
      select(eid, center)
  ) %>% filter(center == 10003) 

fields_sel <- showcase %>%
  filter(Field %in% c("Weight", "Birth weight", "Private healthcare")) %>%
  pull(FieldID)

center_time_sel <- df_all %>%
  select(eid, `20022-0.0`, `4674-0.0`,`21002-0.0`) %>%
  rename("Private healthcare" = `20022-0.0`, "Birth weight" = `4674-0.0`, Weight = `21002-0.0`) %>%
  inner_join(time_center) %>%
  inner_join(center_meaning) %>%
  select(-center, -count) %>%
  rename(center = meaning)


# plot --------------------------------------------------------------------

## raw ---------------------------------------------------------------------
p_all <- missing_rate_histogram(df_all) +
  ggtitle("Overall missing rates") +
  theme(plot.title = element_text(hjust = 0.5)) 

p1_class <- missing_rate_histogram(baseline, color_num = 2) +
  ggtitle("Basic Information") +
  theme(plot.title = element_text(hjust = 0.5)) 

p2_class <- missing_rate_histogram(lifestyle, color_num = 3)+
  ggtitle("Lifestyle") +
  theme(plot.title = element_text(hjust = 0.5))

p3_class <- missing_rate_histogram(measurement, color_num = 4)+
  ggtitle("Measurement") +
  theme(plot.title = element_text(hjust = 0.5))

p4_class <- missing_rate_histogram(environment, color_num = 5)+
  ggtitle("Environment") +
  theme(plot.title = element_text(hjust = 0.5))

p5_class <- missing_rate_histogram(genetic, color_num = 6)+
  ggtitle("Genetic") +
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_x_continuous(limits = c(0, 100))

p_class <- ggarrange(p1_class, p2_class, p3_class, p4_class, p5_class, ncol = 3, nrow = 2)
# p_raw <- ggarrange(p_all, p_class, ncol = 2, labels = c("a", "b"), font.label = list(size = 20))

# center_time -------------------------------------------------------------

p_center_time <- time_center %>%
  inner_join(center_meaning) %>%
  select(-center) %>%
  rename(center = meaning) %>%
  ggplot(., aes(x = time, y = center, fill = center))+
  geom_boxplot() +
  theme_bw() +
  theme(legend.position = "none") +
  xlab("Date of attending assessment center") +
  ylab("Assessment Center") +
  theme(text = element_text(size = 12)) 

## center_time_supp ------------------------------------------------------------------

p_center_time_supp1 <- missing_rate_by_center %>%
  filter(center != "NA") %>%
  ggplot(., aes(y = missing_rate_4598, x = center)) +
  geom_point(color = pal_npg("nrc")(10)[1], size = 3) +
  theme_bw() +
  ylab("missing rate") +
  xlab("Assessment center") +
  geom_segment(aes(xend=center), yend=0, color = pal_npg("nrc")(10)[1], linewidth = 1.5) +
  expand_limits(y=0) +
  theme(axis.text.x = element_text(size = 8, angle = 45, hjust = 1, vjust = 1)) +
  ggtitle("Missing rates of variable \n 'Ever depressed for a whole week'")+
  # theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 12))

p_center_time_supp2 <- var4598_time_center %>%
  filter(center == "11011") %>%
  select(time, `4598-0.0`) %>%
  mutate(missing = as.factor(is.na(`4598-0.0`))) %>%
  mutate(missing = fct_recode(missing, "Missing" = "TRUE", "Observed" = "FALSE")) %>% 
  ggplot(., aes(x = time)) +
  # facet_wrap(~missing, ncol = 1) +
  geom_histogram(aes(y = after_stat(density), fill=missing),color="#e9ecef", alpha=0.6, position = 'identity', bins = 100) +
  geom_density(aes(color = missing), bw = 100, linewidth = 2) +
  # scale_fill_manual(values=c("#69b3a2", "#404080")) +
  theme_bw() +
  xlab("Date of attending assessment center 'Bristol'") + 
  # ylab("missing rate") +
  scale_color_npg() +
  scale_fill_npg() +
  ggtitle("Missing pattern of variable \n 'Ever depressed for a whole week' \n in center 'Bristol'")+
  # theme(plot.title = element_text(hjust = 0.5)) +
  xlim(c(as.Date("2006-01-01"), as.Date("2010-12-31"))) +
  theme(
    legend.position = c(.05, .95),
    legend.justification = c("left", "top"),
  ) +
  theme(text = element_text(size = 12))                    

p_center_time_supp3 <- var4598_time_center %>%
  # filter(center == "11011") %>%
  select(time, `4598-0.0`) %>%
  mutate(missing = as.factor(is.na(`4598-0.0`))) %>%
  mutate(missing = fct_recode(missing, "Missing" = "TRUE", "Observed" = "FALSE")) %>% 
  ggplot(., aes(x = time)) +
  # facet_wrap(~missing, ncol = 1) +
  geom_histogram(aes(y = after_stat(density), fill=missing),color="#e9ecef", alpha=0.6, position = 'identity', bins = 100) +
  geom_density(aes(color = missing), bw = 100, linewidth = 2) +
  # scale_fill_manual(values=c("#69b3a2", "#404080")) +
  theme_bw() +
  xlab("Date of attending assessment center") + 
  # ylab("missing rate") +
  scale_color_npg() +
  scale_fill_npg() +
  ggtitle("Missing pattern of variable \n 'Ever depressed for a whole week'\n")+
  # theme(plot.title = element_text(hjust = 0.5)) +
  xlim(c(as.Date("2006-01-01"), as.Date("2010-12-31"))) +
  theme(
    legend.position = c(.05, .95),
    legend.justification = c("left", "top"),
  ) +
  theme(text = element_text(size = 12))

# pilot -------------------------------------------------------------------

p_pilot_col <- missing_rate_histogram(df_pilot, ymax = 300)
p_pilot_row <- missing_rate_histogram_row(df_pilot)
# p_pilot <- ggarrange(p_pilot_col, p_pilot_row, labels = c("b", "c"), font.label = list(size = 20))  


# center_time_sel ---------------------------------------------------------
p_center_sel <- center_time_sel %>%
  group_by(center) %>% # Grouping by center
  summarise(
    `Private healthcare` = mean(is.na(`Private healthcare`)),
    `Birth weight` = mean(is.na(`Birth weight`)),
    Weight = mean(is.na(Weight))
  ) %>%
  pivot_longer(cols = c(`Private healthcare`, `Birth weight`, Weight)) %>%
  ggplot(., aes(x = name, y = value, fill = center)) +
  geom_bar(stat = "identity",position = "dodge") +
  theme_bw() +
  scale_fill_manual(values = c25, name = "Assessment center") +
  theme(
    legend.position = c(0.99, 0.99),
    legend.justification = c("right", "top"),
    legend.box.just = "right",
    legend.title = element_text(size = 9),
    legend.text = element_text(size = 8),  # Adjust text size,
    legend.key.size = unit(0.5, 'cm')      # Adjust key size
  ) +
  xlab("Variable")+
  ylab("Missing Rate") +
  theme(text = element_text(size = 12)) 

p_time_sel <- center_time_sel %>%
  select(-eid, -center) %>%
  pivot_longer(cols = -time, names_to = "variable") %>%
  mutate(missing = as.factor(is.na(value))) %>%
  mutate(missing = fct_recode(missing, "Missing" = "TRUE", "Observed" = "FALSE")) %>%
  select(-value) %>%
  ggplot(., aes(x = time)) +
  facet_wrap(~variable, ncol = 1)+
  geom_histogram(aes(fill=missing),color="#e9ecef", alpha=0.6, position = 'identity', bins = 40) +
  theme_bw() +
  xlab("Date of attending assessment center") + 
  ylab("Number of participants") +
  scale_color_npg() +
  scale_fill_npg() +
  theme(
    legend.position = "bottom"
  ) +
  theme(text = element_text(size = 12))

p_all ## a
p_class ## b
p_center_sel ## c
p_time_sel ## d
p_center_time ## supp_a
p_pilot_col ## supp_b
p_pilot_row ## supp_c
p_center_time_supp1 ## supp_d
p_center_time_supp2 ## supp_e
p_center_time_supp3 ## supp_f

