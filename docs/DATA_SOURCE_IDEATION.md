# Data Source Ideation — Aggressive Expansion

**Purpose**: Comprehensive catalog of data sources that could improve shift-based community discovery. Organized by signal category. Prioritized by: (1) free/public, (2) county or tract resolution, (3) temporal coverage, (4) expected signal strength for political covariation.

**Currently integrated**: VEST elections, MEDSL elections, ACS demographics, RCMS religion, IRS migration (raw), LODES commuting (raw), TIGER shapefiles, placeholder polls.

---

## Tier 1 — High-Impact, Free, Ready to Fetch

These should be integrated before October 2026. Each has strong theoretical grounding for why it would improve community detection or shift prediction.

### 1. CDC WONDER Mortality Data
- **What**: County-level death rates by cause (heart disease, cancer, drug overdose, suicide, COVID, "deaths of despair")
- **Resolution**: County | **Temporal**: 1999-2023
- **URL**: https://wonder.cdc.gov/
- **Why it matters**: "Deaths of despair" (Case & Deaton) are the strongest county-level predictor of Trump 2016 swing. Opioid mortality rates predict populist shift better than income alone. COVID death rates correlate with political polarization post-2020.
- **Signal**: Mortality patterns capture economic distress, healthcare access, and cultural despair that demographics miss.
- **Cost**: Free, public API

### 2. CDC COVID Vaccination Rates
- **What**: County-level vaccination completion rates, booster uptake
- **Resolution**: County | **Temporal**: 2021-2023
- **URL**: https://data.cdc.gov/Vaccinations/
- **Why it matters**: Vaccination rates became the single strongest county-level correlate of partisan lean by mid-2021 (r > 0.8 with 2020 Trump vote share in many states). This is a revealed-preference political signal that didn't exist before 2021.
- **Signal**: Post-2020 political identity marker. Captures pandemic-era realignment signal that election data alone misses between cycles.
- **Cost**: Free

### 3. FEC Individual Campaign Contributions (Zip-Level)
- **What**: Every individual political contribution >$200, geocoded to zip code. ActBlue/WinRed small-dollar totals.
- **Resolution**: ZIP (aggregatable to county) | **Temporal**: 1980-present, updated daily
- **URL**: https://www.fec.gov/data/browse-data/?tab=bulk-data
- **Why it matters**: Small-dollar donations are a revealed-preference measure of political engagement intensity. The ratio of ActBlue to WinRed contributions is a real-time partisan thermometer at sub-county resolution. Donation velocity (change in contribution rates) is a leading indicator of shift.
- **Signal**: Political engagement intensity + partisan lean at high resolution. Time-varying — captures mobilization waves.
- **Cost**: Free

### 4. BLS QCEW — Industry Composition
- **What**: County x NAICS sector employment and wages, quarterly
- **Resolution**: County | **Temporal**: 1990-present
- **URL**: https://www.bls.gov/qcew/
- **Why it matters**: Industry mix defines economic identity — military bases, university towns, agriculture, manufacturing, tourism, tech, extractive industries all have distinct political profiles. Manufacturing decline vs. service growth is a core realignment driver.
- **Signal**: Economic structure shapes political interests directly. A county that lost 30% of manufacturing jobs between 2012-2020 shifts differently than one that gained healthcare jobs.
- **Cost**: Free (some cells suppressed for small employers)

### 5. Facebook Social Connectedness Index (SCI)
- **What**: County-to-county relative probability of Facebook friendship
- **Resolution**: County-to-county | **Temporal**: ~2020 snapshot
- **URL**: https://data.humdata.org/dataset/social-connectedness-index
- **Why it matters**: Direct measure of "who talks to whom." Bailey et al. (2018) showed SCI predicts economic outcomes, trade flows, and disease spread. For this model: communities that are socially connected should shift together, even if geographically non-adjacent.
- **Signal**: Social network topology for community detection. Could define a "social adjacency" graph alongside spatial adjacency.
- **Cost**: Free (Humanitarian Data Exchange)

### 6. USDA Census of Agriculture
- **What**: Farm count, acreage, crop types, livestock, organic certification, farm income, operator demographics — by county
- **Resolution**: County | **Temporal**: Every 5 years (2017, 2022)
- **URL**: https://www.nass.usda.gov/AgCensus/
- **Why it matters**: Agricultural counties have distinct political behavior. Crop type matters: row crop (corn/soy) counties differ from livestock, specialty crop (fruit/vegetable), and timber counties. Farm consolidation (fewer, larger farms) correlates with rural depopulation and political shift.
- **Signal**: Agricultural economic structure as a community-defining feature. Distinguishes rural subtypes.
- **Cost**: Free

### 7. FHFA House Price Index / Census Building Permits
- **What**: County-level house price appreciation (quarterly HPI) + new residential construction permits
- **Resolution**: County (HPI for ~400 metros + many counties) | **Temporal**: 1991-present
- **URL**: https://www.fhfa.gov/data/hpi, https://www.census.gov/construction/bps/
- **Why it matters**: Housing affordability crisis is a major driver of political attitudes, especially among younger voters. Rapid price appreciation signals gentrification/displacement. Construction permits signal growth pressure.
- **Signal**: Economic mobility proxy. Price change rate may predict shift direction (appreciation → incumbent party advantage? Or → populist backlash from priced-out residents?).
- **Cost**: Free

### 8. BEA Regional Price Parities + Personal Income
- **What**: Cost-of-living adjustment by metro/non-metro areas + county per-capita income decomposition (wages, transfers, investments, government)
- **Resolution**: Metro/state (RPP), County (income) | **Temporal**: 2008-present (RPP), 1969-present (income)
- **URL**: https://www.bea.gov/data/
- **Why it matters**: Nominal income misleads — $50K in rural Alabama vs. Atlanta are different lives. Income composition matters: transfer-dependent counties (Social Security, disability) behave differently from wage-dependent ones. Government transfer dependency is one of the strongest rural-shift predictors.
- **Signal**: Real economic wellbeing + dependency structure.
- **Cost**: Free (BEA API)

### 9. NCES School District Data + EdFacts
- **What**: Per-pupil spending, student-teacher ratios, free/reduced lunch rates, graduation rates, charter/private school enrollment, homeschool rates
- **Resolution**: School district (mappable to county) | **Temporal**: Annual
- **URL**: https://nces.ed.gov/ccd/, https://www2.ed.gov/about/inits/ed/edfacts/
- **Why it matters**: School quality and school choice battles are politically salient. Private/homeschool rates signal cultural conservatism. Free lunch rates are a poverty proxy. Education spending levels reflect local tax base and political preferences.
- **Signal**: Cultural + economic indicator. School choice rates may be the best available proxy for cultural conservatism at the county level.
- **Cost**: Free

### 10. IRS Statistics of Income — Zip-Level Income Data
- **What**: AGI distribution by bracket, EITC claims, mortgage interest deductions, charitable giving — by zip code
- **Resolution**: ZIP code | **Temporal**: Annual
- **URL**: https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-zip-code-data-soi
- **Why it matters**: More granular income distribution than ACS. EITC claim rate is a poverty proxy. Charitable giving rate correlates with religiosity and community engagement. The Gini-like spread of AGI brackets within a zip captures inequality.
- **Signal**: Within-county economic inequality + transfer dependency + civic engagement proxy.
- **Cost**: Free

---

## Tier 2 — Strong Signal, Moderate Integration Effort

### 11. Vera Institute / BJS Incarceration Data
- **What**: County-level jail population, incarceration rates, racial disparities in incarceration
- **Resolution**: County | **Temporal**: 1970-2018
- **URL**: https://github.com/vera-institute/incarceration-trends
- **Why it matters**: Mass incarceration directly affects voter eligibility (felony disenfranchisement varies by state — FL restored rights in 2018 via Amendment 4, then restricted again). Incarceration rates correlate with both racial composition and punitive political culture.
- **Signal**: Criminal justice attitudes + direct voting pool effects. FL-specific: Amendment 4 created a natural experiment.

### 12. FCC Broadband Deployment Data
- **What**: Fixed broadband availability by census block, speed tiers, provider count
- **Resolution**: Census block | **Temporal**: Semi-annual since 2014 (new Broadband Data Collection since 2022)
- **URL**: https://broadbandmap.fcc.gov/data-download
- **Why it matters**: Digital divide is geographic and political. Broadband access affects information ecosystem, remote work capacity, and economic opportunity. Low-broadband areas have different media consumption and political information sources.
- **Signal**: Information environment + economic opportunity proxy. Interacts with WFH migration patterns.

### 13. CES/CCES (Cooperative Election Study) — County-Aggregated Attitudes
- **What**: ~60K respondents/year with validated vote, party ID, issue positions, demographics, geocoded to congressional district/county
- **Resolution**: Individual (aggregatable to county with sufficient N) | **Temporal**: Annual since 2006
- **URL**: https://cces.gov.harvard.edu
- **Why it matters**: The only large-N survey with both validated turnout AND issue-level attitudes at a geographic resolution useful for county estimation. Can produce MRP estimates of issue positions (abortion, guns, immigration attitudes) by county.
- **Signal**: Attitude-level data that election returns can't provide. Why did a county shift? Was it immigration salience? Economic anxiety? Cultural backlash?

### 14. DIME Database — Campaign Finance Ideology Scores
- **What**: CFscore ideology estimates for every candidate and donor, derived from contribution networks (Bonica 2014)
- **Resolution**: Individual candidates/donors | **Temporal**: 1980-2022
- **URL**: https://data.stanford.edu/dime
- **Why it matters**: Measures revealed ideological positioning of candidates and donors. County-level average donor ideology captures the political center of gravity. Shift in average donor ideology over time captures realignment at the activist level.
- **Signal**: Elite/activist ideology positioning. Detects polarization before it shows up in vote shares.

### 15. Opportunity Insights — Social Capital + Economic Mobility
- **What**: County-level social capital index, economic mobility (Raj Chetty), volunteering rates, civic organizations, social cohesion measures
- **Resolution**: County/ZIP | **Temporal**: Various (mostly ~2018-2022)
- **URL**: https://opportunityinsights.org/data/
- **Why it matters**: Chetty's economic mobility data is the gold standard for "American Dream" metrics. Low-mobility counties may shift differently (populist backlash) vs. high-mobility counties (status quo preference). Social capital captures Putnam's "bowling alone" thesis at county resolution.
- **Signal**: Community cohesion + economic mobility. Social capital may moderate or amplify shift magnitude.

### 16. EPA Air Quality + Environmental Justice
- **What**: AQI, PM2.5, ozone levels, EJ screening scores (pollution burden by demographic), Superfund site locations
- **Resolution**: County/tract | **Temporal**: Ongoing
- **URL**: https://www.epa.gov/ejscreen, https://aqs.epa.gov/aqsweb/
- **Why it matters**: Environmental justice communities are politically distinct. Proximity to pollution sources shapes attitudes toward regulation and industry. EPA EJ scores combine environmental burden with demographic vulnerability.
- **Signal**: Environmental politics salience. Counties with high pollution + low income may shift toward regulation-friendly candidates.

### 17. Census Business Dynamics Statistics (BDS)
- **What**: Firm births, deaths, job creation/destruction by county and firm size class
- **Resolution**: County | **Temporal**: 1978-2021
- **URL**: https://www.census.gov/programs-surveys/bds.html
- **Why it matters**: Entrepreneurship rate captures economic dynamism. Net firm births vs. deaths distinguishes growing vs. declining economies. "Company town" counties (dominated by one large employer) behave differently.
- **Signal**: Economic dynamism + dependency. Firm death rates may predict populist shift.

### 18. NOAA Climate Normals + Natural Disaster Exposure
- **What**: Temperature, precipitation, extreme weather frequency + FEMA disaster declarations per county
- **Resolution**: Station/county | **Temporal**: 30-year normals + disaster history
- **URL**: https://www.ncei.noaa.gov/products/land-based-station/us-climate-normals, https://www.fema.gov/api/open/v2/DisasterDeclarations
- **Why it matters**: Climate change exposure varies geographically and affects political attitudes toward climate policy. Repeated hurricane/flood exposure in FL/GA/AL coastal counties may shift attitudes. Agricultural drought affects rural economics.
- **Signal**: Climate vulnerability. Coastal flood risk vs. inland stability is a community-defining feature.

### 19. VA + DoD Military Presence
- **What**: Veteran population percentage, active military installations, military retiree concentration, VA facility locations
- **Resolution**: County | **Temporal**: ACS for veterans, DoD BRAC for installations
- **URL**: ACS B21001 (veterans), https://installations.militaryonesource.mil/
- **Why it matters**: Military communities have distinct political profiles — high veteran concentration correlates with defense-hawkish, socially conservative attitudes, but military base towns are also diverse and federally dependent. Base closures (BRAC) create economic shocks.
- **Signal**: Military culture as a community type. FL (MacDill, Eglin, Pensacola), GA (Fort Eisenhower, Robins), AL (Redstone, Maxwell) all have major installations.

### 20. Local News Desert Map
- **What**: Counties with/without a local newspaper, newspaper circulation trends, local TV station coverage
- **Resolution**: County | **Temporal**: 2004-present
- **URL**: https://www.usnewsdeserts.com/reports/news-deserts-and-ghost-newspapers/
- **Why it matters**: Darr et al. (2018) showed local newspaper closures increase straight-ticket voting and reduce split-ticket behavior. News deserts reduce local political information and increase nationalization of politics — which changes shift patterns.
- **Signal**: Information environment quality. News deserts may exhibit more uniform national-wave shifts; media-rich counties may show more local variation.

---

## Tier 3 — Creative/Unconventional Sources

These require more work or have weaker theoretical grounding, but could capture community signals that traditional sources miss.

### 21. IRS Exempt Organization (990) Data — Nonprofit Density
- **What**: Every 501(c)(3) and 501(c)(4) organization by zip code — type, revenue, mission
- **Resolution**: ZIP | **Temporal**: Annual
- **URL**: https://www.irs.gov/statistics/soi-tax-stats-annual-extract-of-tax-exempt-organization-financial-data
- **Why it matters**: Nonprofit density and type (churches, social services, advocacy, arts) are proxies for civic infrastructure. 501(c)(4) "dark money" organizations signal political mobilization infrastructure.
- **Signal**: Civic infrastructure density + political mobilization capacity.

### 22. USPS Address Vacancy Rate
- **What**: Percentage of addresses receiving no mail (vacant) by zip code
- **Resolution**: ZIP | **Temporal**: Quarterly
- **URL**: HUD Aggregated USPS Administrative Data (https://www.huduser.gov/portal/datasets/usps.html)
- **Why it matters**: Vacancy rates capture population decline, abandonment, and blight better than Census (which only measures every 10 years). Rapidly rising vacancy = community decline.
- **Signal**: Community vitality/decline in near-real-time.

### 23. Ookla/M-Lab Internet Speed Data
- **What**: Actual measured broadband speeds (not just advertised availability from FCC)
- **Resolution**: Tile (aggregatable to county) | **Temporal**: Quarterly
- **URL**: https://www.speedtest.net/insights/blog/best-ookla-open-data-projects/ (Open Data), https://www.measurementlab.net/data/
- **Why it matters**: FCC data measures availability, but actual speeds capture the real digital divide. Speed affects WFH viability, streaming/media consumption, and e-commerce participation.
- **Signal**: Digital infrastructure quality as lived experience.

### 24. Zillow Home Value Index (ZHVI) — Free Tier
- **What**: Median home values and month-over-month/year-over-year appreciation by zip/county
- **Resolution**: ZIP/County | **Temporal**: 2000-present, monthly
- **URL**: https://www.zillow.com/research/data/
- **Why it matters**: Housing wealth is the largest asset class for most Americans. Rapid appreciation creates winners (homeowners) and losers (renters) with divergent political interests. Housing cost burden may drive migration patterns that reshape community composition.
- **Signal**: Wealth effect + displacement pressure.

### 25. EIA Energy Production + Fossil Fuel Employment
- **What**: County-level oil/gas/coal production, renewable energy installations, energy sector employment
- **Resolution**: County | **Temporal**: Annual
- **URL**: https://www.eia.gov/opendata/
- **Why it matters**: Fossil fuel dependent communities have intense opposition to energy transition policies. Renewable energy installation (wind/solar farms) creates new economic interests in rural areas that may moderate traditional resource-extraction politics.
- **Signal**: Energy economy identity. Coal counties, oil counties, and wind-farm counties are different communities.

### 26. FDIC Bank Branch + NCUA Credit Union Data
- **What**: Every bank branch and credit union location with deposits
- **Resolution**: Point (aggregatable to county) | **Temporal**: Annual
- **URL**: https://www.fdic.gov/analysis/quarterly-banking-profile, https://www.ncua.gov/analysis/credit-union-corporate-call-report-data
- **Why it matters**: Bank branch closures correlate with economic decline. Credit union vs. big bank density captures community economic character. "Banking deserts" overlap with news deserts and health deserts.
- **Signal**: Financial infrastructure as community vitality proxy.

### 27. County Health Rankings (Robert Wood Johnson Foundation)
- **What**: Composite health rankings — premature death, health behaviors (smoking, obesity, physical inactivity), clinical care access, social/economic factors
- **Resolution**: County | **Temporal**: Annual since 2010
- **URL**: https://www.countyhealthrankings.org/
- **Why it matters**: Health outcomes are a holistic measure of community wellbeing that integrates economic, social, and environmental factors. Premature death rate captures "deaths of despair" plus general community health infrastructure.
- **Signal**: Community wellbeing composite. May capture latent community types that single-variable measures miss.

### 28. BRFSS (Behavioral Risk Factor Surveillance System)
- **What**: State/metro health behavior survey — smoking, exercise, diet, mental health days, insurance coverage, preventive care
- **Resolution**: State/metro (some county estimates via PLACES) | **Temporal**: Annual
- **URL**: https://www.cdc.gov/brfss/, https://www.cdc.gov/places/
- **Why it matters**: CDC PLACES project produces tract-level modeled estimates for 36 health measures. Health behavior patterns (smoking rates, obesity, mental health) correlate strongly with community type and political lean.
- **Signal**: Behavioral health as cultural marker. Smoking rate is one of the strongest correlates of Trump vote share at county level.

### 29. Voter Registration Party Shifts (FL/GA State Data)
- **What**: Net change in registered Democrats, Republicans, and NPAs by county over time
- **Resolution**: County/precinct | **Temporal**: Monthly (FL), periodic (GA)
- **URL**: FL: https://dos.fl.gov/elections/data-statistics/voter-registration-statistics/
- **Why it matters**: Party registration change is a leading indicator of vote share shift. FL's monthly data allows tracking realignment in near-real-time. The massive FL party registration shift (2016-2024) from D+advantage to R+advantage is a core signal.
- **Signal**: Real-time partisan realignment tracking. Leading indicator for election shift.

### 30. OpenElections Project — Downballot Precinct Returns
- **What**: Precinct-level results for state legislature, county commission, sheriff, school board, ballot measures
- **Resolution**: Precinct | **Temporal**: Growing archive
- **URL**: http://openelections.net/
- **Why it matters**: Downballot races reveal local political dynamics that presidential results mask. Sheriff elections capture law-enforcement attitudes. School board elections capture education culture wars. Ballot measures (abortion, marijuana, minimum wage) reveal issue-specific attitudes.
- **Signal**: Issue-level political behavior. Splits between presidential and downballot voting reveal crossover dynamics.

### 31. SBA Loan Data + Paycheck Protection Program
- **What**: Small business loan approval rates, PPP loan distribution by county
- **Resolution**: ZIP/county | **Temporal**: Annual (SBA), 2020-2021 (PPP)
- **URL**: https://www.sba.gov/funding-programs/loans/lender-match, https://data.sba.gov/
- **Why it matters**: PPP distribution was politically contentious and unevenly distributed. Small business formation/lending rates capture entrepreneurial culture. SBA disaster loan rates after hurricanes capture resilience patterns.
- **Signal**: Economic support dependency + small business culture.

### 32. Google Trends — Relative Search Interest
- **What**: Relative search volume for politically relevant terms by DMA (designated market area)
- **Resolution**: DMA (~210 US markets) | **Temporal**: 2004-present
- **URL**: https://trends.google.com/trends/
- **Why it matters**: Search behavior reveals salience — what issues people are actively seeking information about. "Immigration" searches spike in communities experiencing demographic change. "Gas prices" searches correlate with economic anxiety. Proxy for issue salience without survey overhead.
- **Signal**: Issue salience in near-real-time. DMA resolution is coarser than county but captures media market effects.

### 33. BLS Local Area Unemployment Statistics (LAUS)
- **What**: Monthly unemployment rate by county
- **Resolution**: County | **Temporal**: 1990-present, monthly
- **URL**: https://www.bls.gov/lau/
- **Why it matters**: Monthly resolution allows tracking economic conditions between elections. Unemployment trends (improving vs. worsening) matter more than levels for political behavior. The trajectory of unemployment leading into an election may predict shift direction.
- **Signal**: Economic conditions at high temporal resolution. Change over time is the key feature, not the level.

### 34. USDA SNAP/Food Stamp Participation
- **What**: County-level SNAP participation rates and benefit amounts
- **Resolution**: County | **Temporal**: Annual
- **URL**: https://www.fns.usda.gov/pd/supplemental-nutrition-assistance-program-snap
- **Why it matters**: SNAP participation captures economic distress AND attitudes toward government assistance. High-SNAP counties in the South vote Republican despite receiving federal benefits (the "red state paradox"). Change in SNAP participation may signal economic shock.
- **Signal**: Government dependency + economic distress signal.

### 35. Primary Election Competitiveness + Turnout
- **What**: Primary election results, turnout rates, number of contested primaries by party, by county
- **Resolution**: County/precinct | **Temporal**: Varies by state
- **URL**: FL/GA/AL SOS + OpenElections
- **Why it matters**: Competitive primaries signal active party infrastructure. Low primary turnout in one party suggests disengagement. The ratio of R to D primary voters in FL (where primaries are closed) directly measures party enthusiasm.
- **Signal**: Party infrastructure health + engagement intensity. Leading indicator.

---

## Tier 4 — Speculative / Requires Research

### 36. Yelp/Google Places Business Category Density
- **What**: Density of business types by category — BBQ restaurants, yoga studios, gun shops, organic groceries, Dollar Generals, Walmarts, craft breweries
- **Resolution**: Point (aggregatable) | **Temporal**: Current snapshot
- **Why it matters**: Consumer landscape as cultural proxy. Dollar General density correlates with rural poverty. Craft brewery density correlates with educated urban areas. Gun shop density correlates with gun culture. These are "cultural fingerprints."
- **Signal**: Consumer culture as community type signal. Unconventional but potentially powerful for community detection.
- **Feasibility**: Would need Yelp Fusion API or Google Places API. Rate-limited but feasible for county-level aggregation.

### 37. AM/FM Radio Station Format Data
- **What**: Radio station call signs, formats (country, talk radio, NPR, religious, Spanish-language), coverage areas
- **Resolution**: Station (mappable to coverage area) | **Temporal**: Current
- **URL**: FCC license database
- **Why it matters**: Media environment shapes information exposure. Talk radio density correlates with conservative lean. NPR station density correlates with educated professional communities. Spanish-language radio density captures Hispanic community presence.
- **Signal**: Information/media ecosystem as community signal.

### 38. Church Denomination Locations (Beyond RCMS Aggregates)
- **What**: Individual congregation point locations from ARDA, with denomination
- **Resolution**: Point | **Temporal**: 2020
- **URL**: https://www.thearda.com
- **Why it matters**: Goes beyond county-level RCMS aggregates. Megachurch locations specifically (>2000 weekly attendance) signal a particular type of evangelical community. Mosque/temple/synagogue density captures non-Christian religious communities.
- **Signal**: Fine-grained religious landscape within counties.

### 39. Reddit Community Membership by Geography
- **What**: Subreddit participation correlated with geography (from academic datasets like Baumgartner Reddit corpus)
- **Resolution**: Varies | **Temporal**: 2005-2023
- **Why it matters**: Online community participation reveals cultural affinity. r/guns vs. r/yoga membership by geography is a cultural signal. Political subreddit participation patterns capture engagement intensity.
- **Signal**: Digital culture as community proxy. Academic use only; ethical considerations apply.
- **Feasibility**: Low — requires academic partnership or existing processed datasets.

### 40. NHGIS Historical Census (Deep Time)
- **What**: Census data back to 1790, harmonized to consistent geographic boundaries
- **Resolution**: County | **Temporal**: 1790-2020
- **URL**: https://www.nhgis.org/
- **Why it matters**: Deep historical patterns persist. Counties that were plantation-heavy in 1860 still have distinct political patterns today (Acharya et al., "Deep Roots of Modern Southern Politics"). Historical manufacturing concentration predicts current populism.
- **Signal**: Path dependency. Historical community character may explain current shift patterns that recent data cannot.

---

## Priority Matrix

| Source | Tier | Integration Effort | Expected Signal | Recommend |
|--------|------|-------------------|----------------|-----------|
| CDC WONDER Mortality | 1 | Low (API) | Very High | **DO FIRST** |
| COVID Vaccination Rates | 1 | Low (CSV) | Very High | **DO FIRST** |
| FEC Contributions | 1 | Medium (large files) | High | **DO FIRST** |
| BLS QCEW Industry | 1 | Medium (API) | Very High | **DO FIRST** |
| Facebook SCI | 1 | Low (CSV download) | High | Yes |
| USDA Ag Census | 1 | Medium (API) | High | Yes |
| FHFA HPI / Building Permits | 1 | Low (CSV) | Medium-High | Yes |
| BEA Income + RPP | 1 | Low (API) | Medium-High | Yes |
| NCES School Data | 1 | Medium (multiple files) | High | Yes |
| IRS Zip Income | 1 | Low (CSV) | High | Yes |
| County Health Rankings | 3 | Very Low (single CSV) | High | Yes |
| BLS LAUS Unemployment | 3 | Low (API) | Medium-High | Yes |
| USDA SNAP | 3 | Low (CSV) | Medium | Yes |
| Voter Registration Shifts (FL) | 3 | Medium (scraping) | Very High | **DO FIRST** (FL only) |
| CES/CCES Attitudes | 2 | High (MRP modeling) | Very High | Yes, post-MVP |
| News Desert Map | 2 | Low (CSV) | Medium | Yes |
| VA/Military Presence | 2 | Low (ACS + shapefiles) | Medium-High | Yes |
| Primary Competitiveness | 3 | Medium (state SOS) | High | Yes |
| Google Trends | 3 | Medium (API limits) | Medium | Maybe |
| Business Category Density | 4 | High (API) | Unknown-High | Research first |

---

## Recommended Integration Order (Next 5 Fetchers)

1. **CDC WONDER mortality** — deaths of despair are theoretically the strongest missing signal
2. **COVID vaccination rates** — strongest post-2020 political correlate, trivial to fetch
3. **BLS QCEW industry composition** — already documented as "High" priority, defines economic identity
4. **FL voter registration change** — leading indicator, FL-specific but high resolution
5. **County Health Rankings** — single CSV, composite index, immediate value

After these 5, prioritize: FEC contributions, IRS zip income, Facebook SCI, USDA Agriculture.

---

## Expansion Round 2 — 50 Additional Sources

*Added 2026-03-19. Organized by signal category. Hayden's directive: be aggressive and creative. Think full universe of possibilities.*

---

### NYTimes / Major Media Open Data

### 41. NYT Upshot Precinct-Level Presidential Results (2020)
- **What**: Precinct-polygon GeoJSON with Biden/Trump votes, total votes, vote density per km². Generated by NYT from voter file data where official GIS boundaries weren't available.
- **Resolution**: Precinct | **Temporal**: 2020 presidential
- **URL**: `https://int.nyt.com/newsgraphics/elections/map-data/2020/national/precincts-with-results.geojson.gz` (direct download, ~27 states full coverage)
- **Why it matters**: Precinct-level data is the finest geographic resolution available for election analysis — far below county. Within-county variation (exurban vs. suburban vs. urban precincts) reveals community subtypes invisible at county level. FL, GA both have substantial coverage.
- **Signal**: Within-county political heterogeneity. A county with a precinct range of D+60 to R+80 is structurally different from one uniformly R+20.
- **Cost/Effort**: Free, single download. Requires spatial join to census tracts.

### 42. MIT Election Lab Precinct-Level Returns (2016, 2018, 2020, 2022, 2024)
- **What**: Official precinct-level returns for president, senate, governor, state legislature, and local offices. Standardized schema across states. Maintained by MEDSL.
- **Resolution**: Precinct | **Temporal**: 2016-2024
- **URL**: https://electionlab.mit.edu/data (Harvard Dataverse downloads)
- **Why it matters**: The canonical authoritative source for precinct results — more complete and better-curated than NYT for non-presidential races. Includes 2022 governor and 2024 president critical for the shift vector pipeline. Downballot offices (state legislature, sheriff) capture local political dynamics.
- **Signal**: Fine-grained temporal shift tracking. Precinct-level 2016→2020→2024 shift vectors for community discovery at sub-county resolution.
- **Cost/Effort**: Free, Harvard Dataverse. Large files; state-by-state download.

### 43. FiveThirtyEight Political Elasticity Scores
- **What**: County-level swing/elasticity measures quantifying how sensitive a county is to national political waves. Higher elasticity = more swing-voter composition.
- **Resolution**: County/district | **Temporal**: Derived from 2000-2022 cycles
- **URL**: https://github.com/fivethirtyeight/data/tree/master/political-elasticity-scores
- **Why it matters**: Elasticity is theoretically distinct from partisan lean. A highly elastic county has swing voters who move with national environment; an inelastic county has locked-in base voters who don't respond to wave elections. This is a community-type signal the shift vectors can partially detect but elasticity quantifies directly.
- **Signal**: Voter loyalty vs. volatility. Communities of "swing voters" vs. "base voters" are structurally different in how they respond to candidates.
- **Cost/Effort**: Free, small CSV. Direct from GitHub.

### 44. Washington Post Fatal Police Shootings Database
- **What**: Every fatal police shooting in the US since 2015, with victim race, location, whether armed, mental health status, department name.
- **Resolution**: Point (geocoded address, aggregatable to county) | **Temporal**: 2015-present
- **URL**: https://github.com/washingtonpost/data-police-shootings (CC BY-NC-SA 4.0)
- **Why it matters**: Rate of fatal police shootings per capita is a political lightning rod. Counties with high rates relative to their Black population percentage expose racial justice flashpoints. FL (high rate), GA, AL all have meaningful coverage. Post-George Floyd political realignment is partly explained by local policing patterns.
- **Signal**: Law enforcement culture and racial justice salience. Counties where police kill at high rates may show distinct political dynamics post-2020.
- **Cost/Effort**: Free, CC license. Single CSV download.

### 45. CityLab/Bloomberg Urban-Rural Congressional District Classification
- **What**: Classification of all congressional districts on a 6-category urban-rural spectrum (pure urban, urban-suburban, suburban, sparse suburban, small-town, rural), derived from census block-level population density and commuting patterns.
- **Resolution**: Congressional district (mappable to county via areal interpolation) | **Temporal**: 2018 (with 2022 update)
- **URL**: https://github.com/theatlantic/citylab-data/tree/master/citylab-congress
- **Why it matters**: The urban-suburban-rural spectrum captures a major axis of political identity that simple urban/rural dichotomy misses. "Sparse suburban" and "small-town" categories show distinct political trajectories in the Trump era. FL's I-4 corridor counties (Osceola, Polk) are textbook suburban battleground.
- **Signal**: Urban-rural identity gradient. Community type often tracks urban-rural classification more tightly than income or education.
- **Cost/Effort**: Free, GitHub. CC 4.0 license.

---

### Turnout, Voter Behavior & Election Administration

### 46. USPS Change-of-Address Migration Flows (via HUD)
- **What**: Quarter-over-quarter net migration flows between zip codes derived from USPS address change filings. Separate from IRS migration — captures all moves, not just tax filers.
- **Resolution**: ZIP code | **Temporal**: 2010-present, quarterly
- **URL**: https://www.huduser.gov/portal/datasets/usps.html (HUD aggregation of USPS admin data)
- **Why it matters**: Higher frequency than IRS (quarterly vs. annual), captures renters (who are underrepresented in IRS migration because they don't own property). Shows pandemic-era exurban migration in near-real-time. FL's population growth is primarily driven by in-migration — where from matters politically.
- **Signal**: Population composition change velocity. Counties receiving migrants from Democratic metros (NYC → FL) may be shifting purple from the base.
- **Cost/Effort**: Free, HUD portal. ZIP-level, needs aggregation.

### 47. Early Voting / Absentee Split by County (FL State Data)
- **What**: County-level breakdown of votes cast by method: Election Day in-person, early in-person, mail/absentee — split by registered party.
- **Resolution**: County | **Temporal**: 2004-present (FL), 2016-present (GA)
- **URL**: FL: https://dos.fl.gov/elections/data-statistics/elections-data/voter-turnout/ | GA: Secretary of State
- **Why it matters**: Voting method became a partisan proxy post-2020 — Republicans shifted toward Election Day, Democrats toward mail. Party-specific early voting rates signal partisan mobilization infrastructure quality. Counties where the GOP turned out via Election Day vs. mail show different ground-game strategies.
- **Signal**: Partisan mobilization patterns and election administration exposure. Method splits also expose counties to mail ballot rejection rate disparities.
- **Cost/Effort**: Free, state SOS downloads. FL most complete; GA improving.

### 48. Ballot Rejection and Cure Rates (State Data)
- **What**: Rate of mail ballots rejected (due to signature mismatch, missing witness, late arrival) and successfully cured by county. Rejection rates disproportionately affect certain demographics.
- **Resolution**: County | **Temporal**: 2018-present (FL, GA)
- **URL**: FL Div. of Elections, GA Secretary of State; also academic compilations at Harvard Kennedy School Voting Rights Data
- **Why it matters**: Ballot rejection is a direct suppression signal. FL's 2018 Senate race was decided by ~10K votes while ~25K ballots were rejected. High rejection rates in specific counties (often with large Black or Hispanic populations) signal election access barriers that shape political engagement.
- **Signal**: Election access inequity as community-defining feature. Counties with high rejection rates may show depressed Democratic vote share for structural reasons.
- **Cost/Effort**: Free but requires manual state SOS downloads. Some years require FOIA requests.

### 49. Voter Roll Purge Data (Brennan Center / State SOS)
- **What**: Number of voters removed from rolls by county and purge reason (moved, died, inactivity, felony conviction).
- **Resolution**: County | **Temporal**: 2012-present
- **URL**: Brennan Center for Justice annual purge studies + FL/GA/AL SOS raw files
- **Why it matters**: FL purged 340K voters before the 2000 election (the "butterfly ballot" and felon list controversy). GA's aggressive purge activity under Kemp (SOS while running for governor) is a key context for 2018 Abrams-Kemp race. Purge rate is a direct measure of voter suppression intensity and political contestation over ballot access.
- **Signal**: Electoral participation access. Counties with high purge rates may have systematically different electorate compositions than registration numbers suggest.
- **Cost/Effort**: Free. Brennan Center aggregations available as PDFs; state SOS files sometimes available as spreadsheets.

### 50. Redistricting Competitiveness Index (Dave's Redistricting App / Princeton Gerrymandering Project)
- **What**: District-level partisan composition under current and historical maps; gerrymandering efficiency gap and bias scores; competitive district count by state.
- **Resolution**: District (county mappable) | **Temporal**: Per redistricting cycle (2010, 2020)
- **URL**: https://davesredistricting.org/, https://gerrymander.princeton.edu/
- **Why it matters**: Gerrymandering shapes which communities have competitive elections vs. safe seats. Counties packed into safe R or D districts have different political dynamics from competitive swing-district counties. FL's extreme gerrymandering post-2022 (DeSantis-drawn maps) is a major structural shift in which communities matter.
- **Signal**: Electoral competitiveness exposure. Safe-seat counties may show different turnout and shift patterns than competitive districts.
- **Cost/Effort**: Free. DRA has exportable JSON; Princeton publishes CSVs.

---

### Economic Granular Signals

### 51. Trade Adjustment Assistance (TAA) Certifications by County
- **What**: County-level counts of workers certified as job-displaced due to import competition or offshoring under the federal TAA program. Each certification = verified trade-displaced workers.
- **Resolution**: County | **Temporal**: 1975-present (public petitions database)
- **URL**: https://www.dol.gov/agencies/eta/tradeact/taa/taa_search_form.cfm (petition database) + DOL annual reports
- **Why it matters**: Dorn, Autor, Hanson (2013) showed counties more exposed to Chinese import competition shifted toward extreme parties. TAA certifications are a direct, verified measure of trade displacement — not just exposure. AL and rural GA had significant textile/apparel TAA certifications through the 1990s-2000s.
- **Signal**: Verified trade displacement trauma as a community-defining experience. The "China shock" counties are a specific community type with documented political effects.
- **Cost/Effort**: Free, searchable database. Requires scraping/aggregation to county level.

### 52. USDA Rural-Urban Continuum Codes (Beale Codes)
- **What**: 9-category county classification on rural-urban spectrum: metro by population size (1-3), adjacent/non-adjacent micropolitan (4-7), rural (8-9).
- **Resolution**: County | **Temporal**: 1974, 1983, 1993, 2003, 2013, 2023 (every ~10 years)
- **URL**: https://www.ers.usda.gov/data-products/rural-urban-continuum-codes/
- **Why it matters**: Far richer than metro/non-metro binary. Distinguishes counties adjacent to large metros (often politically purple bedroom communities) from isolated small-city and truly rural counties. 2003 vs. 2013 code changes reveal counties reclassified as more or less urban — a structural change signal.
- **Signal**: Rural-urban identity gradient at fine resolution. Code change = community structural shift. Adjacent-rural counties (codes 4,6) are often the key swing communities in FL/GA/AL.
- **Cost/Effort**: Free, single Excel download. Already well-documented; trivial to integrate.

### 53. Amazon Warehouse / Fulfillment Center Locations
- **What**: Addresses and approximate employment levels of Amazon fulfillment centers, sortation centers, delivery stations (DSP), and data centers.
- **Resolution**: Point (aggregatable to county) | **Temporal**: 2012-present (point snapshots)
- **URL**: MWPVL International tracking (https://mwpvl.com/html/amazon.html), enhanced by Indeed/LinkedIn job postings
- **Why it matters**: Amazon warehouses are major employment anchors in rural and exurban counties (e.g., Lakeland FL, Macon GA). They employ large numbers of workers at wages above local retail but below manufacturing. They also bring racial diversity to previously homogeneous areas. The "Amazon effect" on local retail closure is a political economy signal.
- **Signal**: Economic disruption and opportunity simultaneously. Counties landing Amazon facilities may shift in complex ways — new jobs vs. displacement of local retail. AL's recent Amazon buildup is politically understudied.
- **Cost/Effort**: Free (MWPVL tracks all facilities). Manual data compilation; ~1,000 facilities nationwide.

### 54. Dollar Store Density (Dollar General, Dollar Tree, Family Dollar)
- **What**: Location counts and density of Dollar General, Dollar Tree, and Family Dollar stores by county.
- **Resolution**: County (from point locations) | **Temporal**: Current + historical via archived store locators
- **URL**: Store locators + OpenStreetMap + SafeGraph historical patterns
- **Why it matters**: Dollar store density is among the strongest predictors of deep poverty in rural areas — Dollar General explicitly targets areas underserved by grocery stores (food deserts). Dollar General's aggressive rural expansion (~20K stores as of 2026) is a direct measure of retail poverty geography. This is a more current and sensitive poverty indicator than SNAP.
- **Signal**: Deep retail poverty geography. Dollar General presence > Walmart presence signals the bottom of the rural economic hierarchy. Rural AL, GA counties with DG density are a distinct community type.
- **Cost/Effort**: Free via OSM or SafeGraph community data. Some point data requires API access.

### 55. Gig Economy Worker Density (DoorDash / Lyft / Instacart Activity)
- **What**: Estimated gig worker concentration by county, derived from BLS "contingent worker" supplements, academic gig economy research datasets, and Uber/Lyft activity estimates.
- **Resolution**: County (estimated) | **Temporal**: 2017-present
- **URL**: BLS Contingent Worker Supplement (2017, 2021): https://www.bls.gov/cps/contingent-and-alternative-employment-arrangements-faqs.htm | Academic: Katz & Krueger (Princeton), JPMorgan Chase Institute gig income data
- **Why it matters**: Gig workers lack union protections, employer-sponsored benefits, and stable income — a distinct economic precariat. Gig economy participation is high in both urban (delivery, rideshare) and rural areas (TaskRabbit, care work). The political attitudes of gig workers differ from traditional employees.
- **Signal**: Labor precarity and economic insecurity without the "blue collar" identity of manufacturing workers. May predict distinct political behavior — particularly in urban exurbs with high delivery worker density.
- **Cost/Effort**: Free for BLS data. JPMorgan Chase Institute data requires academic access request.

### 56. EPA Energy Burden / Low-Income Energy Affordability Data (LEAD Tool)
- **What**: Census-tract-level energy burden (% of household income spent on energy bills), annual energy cost, and household income for low-income populations.
- **Resolution**: Census tract | **Temporal**: 2018 base year (updated 2022)
- **URL**: https://www.energy.gov/scep/slsc/lead-tool (DOE LEAD Tool, downloadable CSV)
- **Why it matters**: High energy burden (>6% of income) is a poverty signal concentrated in the rural South — exactly FL/GA/AL. It captures both housing inefficiency (old/mobile homes) and income inadequacy. Energy insecurity correlates with both economic distress and political attitudes toward utility regulation and fossil fuels.
- **Signal**: Energy poverty as a community marker. Distinct from absolute income — energy burden reveals structural vulnerability of specific housing stock and climate conditions (high AC demand in FL/AL/GA summers).
- **Cost/Effort**: Free, DOE portal. Tract-level CSV download, nationwide.

---

### Cultural & Consumer Signals

### 57. Vehicle Registration Types — Trucks/SUVs vs. EVs vs. Sedans
- **What**: County-level vehicle registration counts by vehicle type category: pickup trucks, SUVs, passenger cars, and electric/hybrid vehicles.
- **Resolution**: County | **Temporal**: Annual
- **URL**: State DMVs (FL HSMV, GA DOR, AL DOR) + Atlas EV Hub for EV subset; FHWA HPMS for aggregate vehicle counts
- **Why it matters**: Pickup truck ownership is among the strongest cultural markers of rural conservative identity. The truck-to-car ratio at the county level is a political signal that pre-dates MAGA. Meanwhile, EV registration concentration in coastal FL and Atlanta suburbs maps directly onto the college-educated professional community type. These are revealed-preference cultural signals without survey overhead.
- **Signal**: Cultural identity revealed through consumption. FL's EV adoption (high in Miami-Dade, Broward, Palm Beach, low in Panhandle) mirrors the political map almost perfectly.
- **Cost/Effort**: Free from state DMVs (some require FOIA). Atlas EV Hub (atlasevhub.com) aggregates EV data for FL and other states.

### 58. Library Circulation and Usage Statistics (IMLS Public Libraries Survey)
- **What**: Annual per-capita library visits, circulation, electronic resource usage, library funding, branch count, and hours of operation by county-equivalent library system.
- **Resolution**: Library system (~county) | **Temporal**: Annual since 1988
- **URL**: https://www.imls.gov/research-evaluation/data-collection/public-libraries-survey (IMLS, free CSV download)
- **Why it matters**: Library usage rate is a community investment and civic engagement signal. Counties where libraries are well-funded and heavily used have different civic cultures from those where libraries are chronically underfunded. Library funding battles are a proxy for culture war intensity (book banning, LGBTQ collections). FL and TX library systems have been under specific political attack since 2022.
- **Signal**: Civic infrastructure investment and cultural conflict. Low circulation + low funding + book banning activity = a specific type of culture war community.
- **Cost/Effort**: Free, IMLS. Clean CSV, nationwide coverage, trivial to integrate.

### 59. National Survey of Fishing, Hunting, and Wildlife-Associated Recreation
- **What**: County or state-level estimates of hunting and fishing license holders, days spent hunting/fishing, and wildlife watching participants.
- **Resolution**: State (some county estimates) | **Temporal**: Every 5 years (2011, 2016, 2022)
- **URL**: https://wsfrprograms.fws.gov/subpages/nationalsurvey/national_survey.htm (USFWS)
- **Why it matters**: Hunting license rates are a strong proxy for rural, white, male cultural identity — the core of the Trump coalition. States and counties with high hunting participation have distinct gun culture, land use attitudes, and wildlife policy preferences. FL's hunting is concentrated in North Florida (Panhandle, rural central) — exactly where political culture differs most from South Florida.
- **Signal**: Outdoor/hunting culture as community identity marker. Hunting license density predicts Republican lean more robustly than income in rural areas.
- **Cost/Effort**: Free, USFWS. State-level primary; county estimates sometimes available from state wildlife agencies (FL FWC).

### 60. Spotify / Music Streaming Genre Geography
- **What**: Relative popularity of music genres (country, hip-hop, Christian/gospel, reggaeton, classic rock) by metro area or state, from Spotify streaming data.
- **Resolution**: Metro area/DMA | **Temporal**: Annual (Spotify Insights)
- **URL**: Spotify for Developers (aggregate data via academic partnerships), plus academic papers using Spotify API data (e.g., Rentfrow's "Do Re Mi" AAAS 2013)
- **Why it matters**: Rentfrow et al. showed music preferences predict personality traits at the county level — country music correlates with conscientiousness and conservatism; hip-hop correlates with openness. Genre geography is a cultural signal available at fine resolution without survey overhead. Spanish-language music concentration directly identifies Hispanic communities.
- **Signal**: Cultural identity through revealed music preferences. May capture community types that demographic variables describe poorly — the "country music belt" in North FL/South GA/AL is a political identity as much as a genre.
- **Cost/Effort**: Free Spotify API for aggregate genre data; academic data partnership for fine-grained geographic analysis.

### 61. NCAA Sports Fan Geography / College Football Fandom
- **What**: Geographic distribution of college football team fandom by county, derived from Facebook/Twitter followership data or sports analytics firms.
- **Resolution**: County | **Temporal**: ~2020 snapshot
- **URL**: NYT/Upshot sports fandom maps (published 2014 for NFL, college), Twitter API academic access, Morning Consult team polling
- **Why it matters**: College football fandom in the Deep South (FL/GA/AL) is a proxy for community identity that transcends race and income — it's one of the few cross-racial community bonds. Alabama's iron bowl geography (Auburn vs. Alabama fans) maps onto interesting county-level cultural fault lines. SEC fandom signals regional identity.
- **Signal**: Community identity through sports affiliation. The Gator/Seminole/Hurricane divide in FL maps partly onto political geography. The SEC belt has distinct political culture from non-SEC regions.
- **Cost/Effort**: Limited free data (NYT published static maps); Twitter API data requires scraping. Morning Consult publishes periodic team popularity surveys by state.

### 62. Gun Permit / Concealed Carry License Rates
- **What**: County-level concealed carry permit (CCP) issuance rates, applications, and active license holders.
- **Resolution**: County | **Temporal**: Annual
- **URL**: FL Dept. of Agriculture and Consumer Services (publishes CCP by county); GA Attorney General; AL Sheriff's Assoc.; also Crime Prevention Research Center tracks nationally
- **Why it matters**: Concealed carry rate is a direct, legally-recorded measure of gun culture and self-defense identity. FL's shall-issue CCP system means permit rates reflect cultural demand, not bureaucratic gatekeeping. FL has ~2.6M active CCPs (highest per-capita state). County-level variation within FL is dramatic — North FL counties have 3-4x higher rates than South FL.
- **Signal**: Gun culture intensity as community identity. CCP rate predicts Republican vote share at county level almost as well as religious attendance. Key for distinguishing rural North FL (high CCP, low income) from rural South FL (agricultural, Hispanic, lower CCP).
- **Cost/Effort**: Free, state agency downloads. FL most transparent; AL requires FOIA for some counties.

### 63. Farmers Market and Direct-Farm Sales Density
- **What**: Number of farmers markets, CSA operations, and farm stands per county, from USDA Local Food Directories and Census of Agriculture direct sales data.
- **Resolution**: County | **Temporal**: Annual (markets), 5-year (ag census)
- **URL**: https://www.ams.usda.gov/local-food-directories/farmersmarkets (USDA API available); USDA Census of Agriculture Table 2 (direct sales)
- **Why it matters**: Farmers market density is a strong proxy for college-educated, health-conscious, economically comfortable community — the "Whole Foods voter" archetype. In FL, farmers market concentration in Gainesville, Sarasota, St. Pete maps directly onto the purple-to-blue county types. Rural farmers markets signal different community types (local food economy resilience) than urban ones.
- **Signal**: Cultural and economic community type — the artisanal/organic food economy as political identity marker. Distinct from Walmart/Dollar General axis.
- **Cost/Effort**: Free, USDA API. Point data needs county aggregation. Census of Agriculture direct sales is already county-level.

---

### Infrastructure & Access Deserts

### 64. Rural Hospital Closures (Sheps Center / CMS Data)
- **What**: County-level record of rural hospital closures since 2005 — hospital name, closure year, bed count, service type remaining. Maintained by UNC Sheps Center from CMS provider enrollment data.
- **Resolution**: Hospital point / County | **Temporal**: 2005-present (195 closures tracked)
- **URL**: https://www.shepscenter.unc.edu/programs-projects/rural-health/rural-hospital-closures/ (downloadable dataset)
- **Why it matters**: Rural hospital closure is one of the most direct measures of rural healthcare collapse — a key driver of "deaths of despair" and political frustration. AL and GA have among the highest rural closure rates nationally; FL's tourist economy has differential urban/rural hospital access. Closure creates a 30-60 minute emergency travel time increase — directly affects maternal mortality and heart attack survival.
- **Signal**: Healthcare infrastructure collapse as community trauma. Counties that lost their only hospital in the last decade are experiencing a specific type of rural decline that shifts political attitudes.
- **Cost/Effort**: Free, Sheps Center downloadable dataset. Single CSV. Very low integration effort.

### 65. HRSA Health Professional Shortage Areas (HPSA) and Medically Underserved Areas (MUA)
- **What**: Federal designation of areas lacking sufficient primary care, mental health, and dental care providers. Also includes Medically Underserved Populations (MUP) designations.
- **Resolution**: County/sub-county geographic area or facility | **Temporal**: Continuously updated
- **URL**: https://data.hrsa.gov/tools/shortage-area/hpsa-find (HRSA Data Warehouse, GIS downloads available)
- **Why it matters**: HPSA designation = federal recognition of healthcare provider shortage. Mental health HPSAs are particularly relevant — they map closely onto opioid crisis counties and "deaths of despair" geography. FL Panhandle, rural GA, and nearly all of rural AL are blanketed with HPSA designations. This is a direct measure of healthcare access desert.
- **Signal**: Healthcare access deprivation as community stressor. Mental health shortage areas may be the best available proxy for latent opioid/despair vulnerability before mortality data shows up.
- **Cost/Effort**: Free, HRSA GIS shapefile download. Requires spatial aggregation to county.

### 66. Childcare Desert Index (CAP / USDA ERS)
- **What**: Ratio of children to licensed childcare capacity by county — counties with >3 children per licensed childcare slot are "childcare deserts."
- **Resolution**: County | **Temporal**: 2019, 2023 (Center for American Progress), ongoing (USDA ERS)
- **URL**: Center for American Progress: https://www.americanprogressaction.org/childcare-data-download/ | USDA ERS rural childcare access reports
- **Why it matters**: Childcare desert concentration shapes labor force participation (especially women's workforce participation), political attitudes toward childcare policy, and economic opportunity. Rural FL/GA/AL are heavily childcare-scarce. Post-pandemic childcare collapse is a major driver of female labor force exit — which has direct electoral implications.
- **Signal**: Family economic stress and women's economic opportunity. Counties in childcare deserts may show distinct voting patterns among parents of young children — one of the most politically mobile demographic segments.
- **Cost/Effort**: Free, CAP downloadable data. County-level, manageable size.

### 67. Pharmacy Desert Map (RUPRI / NCPA)
- **What**: Distance to nearest retail pharmacy by county, pharmacy closure rate since 2010, pharmacy deserts (rural areas with no pharmacy within 10 miles).
- **Resolution**: County | **Temporal**: 2010-present
- **URL**: Rural Policy Research Institute (RUPRI) pharmacy access analyses; NCPA (National Community Pharmacies Assoc.) closure tracking; CMS provider data for pharmacy locations
- **Why it matters**: Pharmacy deserts overlap with hospital deserts and news deserts. Access to prescription medications (including insulin, blood pressure drugs) is a healthcare access measure that particularly affects the elderly rural population — a politically active demographic. AL has the highest rural pharmacy closure rate of any southern state.
- **Signal**: Healthcare retail infrastructure collapse. Pharmacy desert counties have older, sicker, more healthcare-dependent populations with direct personal stakes in healthcare policy.
- **Cost/Effort**: Free (RUPRI publishes county-level analyses as PDFs, sometimes CSV). Some compilation required.

### 68. Public Transit Availability Index (APTA / FTA)
- **What**: County-level transit availability score: presence of fixed-route bus, commuter rail, light rail; service frequency; transit ridership per capita.
- **Resolution**: County | **Temporal**: Annual (NTD data)
- **URL**: FTA National Transit Database: https://www.transit.dot.gov/ntd/ntd-data | APTA transit agency locator
- **Why it matters**: Car dependency is both an economic burden and a political identity marker. Zero-transit counties have different labor markets (longer commutes, worse access to jobs) and different political cultures than transit-accessible areas. In FL, the I-4 corridor's transit void (despite density) contrasts with Miami's system; GA has MARTA but rural GA has nothing.
- **Signal**: Car-dependency as economic and cultural identity. Transit access stratifies who can access jobs, healthcare, and community — with direct political implications for attitudes toward urban policy.
- **Cost/Effort**: Free, FTA NTD. Some aggregation from agency to county level needed.

### 69. Food Desert / Food Access Atlas (USDA ERS)
- **What**: Census-tract-level measure of low-income populations more than 0.5/1/10 miles from nearest supermarket. Includes food swamp measures (ratio of fast food/convenience stores to healthy options).
- **Resolution**: Census tract | **Temporal**: 2015, 2019 (updates ongoing)
- **URL**: https://www.ers.usda.gov/data-products/food-access-research-atlas/ (downloadable Excel/CSV)
- **Why it matters**: Food access geography maps closely onto poverty and race in the South. Rural AL and GA Black Belt counties have among the highest food desert concentrations in the nation. Food deserts create health burdens (diabetes, obesity) that then show up in political attitudes toward healthcare and government assistance. "Food swamp" measure (fast food density) is distinct from food desert — an urban-specific poverty signal.
- **Signal**: Food access equity as community health and economic stressor. Black Belt food desert concentration is a structural feature of the plantation economy's legacy.
- **Cost/Effort**: Free, USDA ERS. Excel download, tract-level. Very manageable.

---

### Environmental Signals

### 70. PFAS Contamination in Drinking Water Systems
- **What**: Public water system-level PFAS detection results under UCMR 3 (2013-2015), UCMR 5 (2023-2025), and EPA's new Maximum Contaminant Level enforcement. Number of systems with exceedances, per-county.
- **Resolution**: Water system (aggregatable to county) | **Temporal**: 2013-present
- **URL**: EPA UCMR data: https://www.epa.gov/dwucmr/data-download-sixth-unregulated-contaminant-monitoring-rule | EWG PFAS map: https://www.ewg.org/interactive-maps/pfas_contamination/
- **Why it matters**: PFAS contamination is concentrated around military bases (Pensacola/Eglin FL, Fort Benning/Bragg GA) due to AFFF firefighting foam use. It also appears around industrial sites (chemical manufacturing, textile plants) — concentrated in rural GA/AL. Communities facing PFAS contamination have specific political grievances against both the military and chemical industry regulators.
- **Signal**: Environmental justice flashpoint — military and industrial PFAS sources create communities with specific federal government grievances. Pensacola FL PFAS from Eglin/NAS Pensacola is a documented case.
- **Cost/Effort**: Free, EPA download. Requires spatial join from water system to county.

### 71. FEMA National Flood Insurance Program (NFIP) Policy and Claims Data
- **What**: County-level NFIP policy count, total insured value, claims paid, repetitive loss properties. Separate from disaster declarations — ongoing insurance penetration and loss experience.
- **Resolution**: County | **Temporal**: Ongoing (annual)
- **URL**: https://www.fema.gov/about/open/data (FEMA OpenFEMA bulk download)
- **Why it matters**: FL has 40% of all US NFIP policies — flood risk is existential for coastal communities. High repetitive loss counties (low-lying coastal areas) face economic threats from FEMA rate increases and sea level rise that directly affect property values and community viability. Attitudes toward climate adaptation vs. denial correlate with flood risk exposure.
- **Signal**: Climate risk as community defining feature. FL coastal counties with high NFIP concentration have specific economic stake in sea level rise and insurance availability — a political signal distinct from inland counties.
- **Cost/Effort**: Free, FEMA OpenFEMA. County-level bulk CSV.

### 72. Wildfire Risk Index (USFS SILVIS Lab / FSim)
- **What**: County and census tract wildfire hazard potential scores, combining fuel load, weather patterns, and topography. Annual burn area and structure loss.
- **Resolution**: 30m raster (aggregatable to county/tract) | **Temporal**: Annual (fire season data), 30-year normals
- **URL**: USFS Wildfire Hazard Potential: https://www.fs.usda.gov/rds/archive/Catalog/RDS-2020-0016-2 | National Interagency Fire Center: https://www.nifc.gov/fire-information/statistics
- **Why it matters**: While FL/GA/AL don't face the extreme wildfire risks of the West, FL panhandle pine flatwoods, GA Okefenokee region, and AL Bankhead Forest areas have significant wildfire exposure. Prescribed burn policy is politically contested (landowner rights vs. wildfire management). Smoke events affect agriculture and health.
- **Signal**: Land management attitudes and rural property rights politics. Wildfire/prescribed burn counties have specific policy grievances. Less relevant for FL/GA/AL than Western states but non-trivial.
- **Cost/Effort**: Free, USFS raster data. Requires GIS processing to aggregate to county.

### 73. Coal Mine / Power Plant Retirement Map (EIA + Sierra Club)
- **What**: Point locations and employment of operating and retired coal mines, coal power plants, and natural gas plants. Retirement dates and worker counts.
- **Resolution**: Plant/mine point | **Temporal**: 1985-present
- **URL**: EIA Power Plant data: https://www.eia.gov/electricity/data/eia923/ | Sierra Club Beyond Coal tracker | Mine Safety and Health Administration (MSHA) mine data
- **Why it matters**: AL has significant coal employment (Walker County, Jefferson County coalfields). Plant closures create "stranded communities" with specific economic trauma and resentment toward environmental regulation. The link between coal country and Trump realignment is one of the most documented political economy findings of the 2010s.
- **Signal**: Industrial transition trauma. Coal-adjacent counties in AL show distinct shift patterns reflecting deindustrialization resentment. Even in non-coal FL/GA, natural gas plant placement creates economic dependency.
- **Cost/Effort**: Free, EIA and MSHA downloads. Point data needs county aggregation.

### 74. USDA Conservation Reserve Program (CRP) Enrollment
- **What**: Acres of agricultural land removed from production and placed in conservation easements, by county. Also includes Wetland Reserve Program (WRP) and other USDA conservation program enrollment.
- **Resolution**: County | **Temporal**: Annual
- **URL**: USDA Farm Service Agency: https://www.fsa.usda.gov/programs-and-services/conservation-programs/reports-and-rankings/ (county-level spreadsheets)
- **Why it matters**: CRP enrollment is a direct measure of agricultural economic marginal land — land that farmers are paid not to farm. High CRP counties are struggling agricultural communities where government payments substitute for market income. It also captures conservation attitudes (farmers who voluntarily participate vs. those who don't).
- **Signal**: Agricultural decline and government dependency. CRP-heavy counties may be experiencing the last generation of farming — a community existential stress that drives political anger. FL Panhandle and SW Georgia have significant CRP acres.
- **Cost/Effort**: Free, USDA FSA. Annual county spreadsheets.

---

### Historical & Structural Signals

### 75. HOLC Redlining Maps (Mapping Inequality / Richmond Fed)
- **What**: Digitized Home Owners' Loan Corporation (1935-1940) neighborhood security maps ("redlining") for 239 US cities, with grade (A-D) polygons. Overlapping with current census tracts to measure legacy effects.
- **Resolution**: Neighborhood polygon (HOLC zone, mappable to census tract) | **Temporal**: 1935-1940 (historical); legacy effects in current data
- **URL**: https://dsl.richmond.edu/panorama/redlining/ (Digital Scholarship Lab, University of Richmond; GeoJSON download)
- **Why it matters**: Redlining created durable wealth gaps that persist 80+ years later. Nelson et al. (2019) showed HOLC grades predict current demographics, health outcomes, and tree canopy almost perfectly. For FL/GA/AL: Jacksonville, Miami, Atlanta, Birmingham, Mobile all have mapped HOLC zones. Redlined areas show specific patterns of racial wealth gap, political participation, and structural disadvantage.
- **Signal**: Structural racism legacy as community-defining historical force. Tract-level HOLC grade is among the most powerful predictors of current racial wealth disparities and political engagement gaps in Black communities.
- **Cost/Effort**: Free, Digital Scholarship Lab GeoJSON. Requires spatial join to census tracts.

### 76. Historical Slave Population (1860 Census)
- **What**: County-level slave and free population from the 1860 Census, including cotton/tobacco/rice production. Via NHGIS (already in list) or Kaggle cleaned versions.
- **Resolution**: County (historical boundaries, need to crosswalk to current) | **Temporal**: 1860
- **URL**: IPUMS NHGIS historical tables; also Acharya, Blackwell, Sen's "Deep Roots" replication data at Harvard Dataverse
- **Why it matters**: The single strongest predictor of modern white Democratic voting in the South is 1860 slave population share (Acharya, Blackwell & Sen 2018 — "Deep Roots of Modern Southern Politics"). Counties with high historical slave populations have more racially polarized politics today. The Black Belt across AL/GA/MS is named for plantation topsoil — the political and agricultural geography are the same.
- **Signal**: Path dependency — 160-year-old slavery geography predicts modern political polarization. The mechanism is white racial resentment sustained by post-Reconstruction political institutions.
- **Cost/Effort**: Free via NHGIS or Acharya et al. replication data. Low integration effort; the key variable (pct_slave_1860) is a single column.

### 77. Sundown Town Database (Tougaloo College / James Loewen)
- **What**: County and municipality-level database of historically documented "sundown towns" — places that excluded Black residents through law, violence, or custom after dark.
- **Resolution**: Municipality/county | **Temporal**: Historical (most active 1890-1968)
- **URL**: https://justice.tougaloo.edu/ (Tougaloo College database, ~10,000 entries nationally)
- **Why it matters**: Sundown towns created highly racially homogeneous white communities with particular historical attitudes toward racial mixing and civil rights. Their demographic legacy persists. Northern FL and southern GA have documented sundown town histories. This is a distinct historical structural variable from the plantation slavery axis.
- **Signal**: Historical white racial exclusion as community structural feature. Sundown town legacy may explain why certain rural white communities have particularly strong racial anxiety signals even in counties with low current Black population.
- **Cost/Effort**: Free, Tougaloo database. Requires geocoding of municipality names to FIPS. Some manual validation needed.

### 78. Historical Union Density (BLS Unionstats / CPS)
- **What**: State-level union membership rates from Barry Hirsch/David Macpherson's "Union Membership and Coverage Database" (unionstats.com), derived from Current Population Survey. Historical series back to 1964.
- **Resolution**: State (some metro areas) | **Temporal**: 1964-present
- **URL**: http://www.unionstats.com/ (Hirsch-Macpherson database, free download)
- **Why it matters**: Union density collapse is a key mechanism of the working-class political realignment (Leighley & Nagler 2007). States that were highly unionized in 1970 but experienced rapid union decline show the sharpest working-class Democratic → Republican shift. AL's steelworker union legacy vs. right-to-work passage creates a natural experiment. FL is a right-to-work state with historically low union density.
- **Signal**: Labor institutional legacy. Historical union density predicts where working-class politics was once organized around class identity vs. cultural identity. Its collapse predicts the timing of realignment.
- **Cost/Effort**: Free, unionstats.com annual downloads. State-level; requires assumptions to distribute to county.

### 79. NLRB Union Election Results by County
- **What**: Every union representation election filed with NLRB — employer name, location, election result, union, vote counts — since 1970. Includes recent Amazon/Starbucks unionization campaigns.
- **Resolution**: Employer/workplace (aggregatable to county) | **Temporal**: 1977-present
- **URL**: https://www.nlrb.gov/resources/data/graphs-data (NLRB case management system; bulk data available via FOIA or academic datasets)
- **Why it matters**: Recent union election wave (2021-2024 at Amazon, Starbucks, Apple) is concentrated in specific counties. Counties where workers are currently attempting to unionize are sites of active labor conflict — a political economy signal. Historical NLRB data shows where labor organizing was and is active, separate from membership (which includes inactive unions).
- **Signal**: Active labor organizing as community political economy signal. Amazon warehouse counties with union votes may be developing a distinct working-class political identity.
- **Cost/Effort**: Free, NLRB website. Some data requires FOIA; academic datasets (NLRB Elections Database from Cornell ILR) available.

### 80. Lynching Victims Data (Equal Justice Initiative / Monroe Work Project)
- **What**: County-level counts of racial lynching victims from 1877-1950, from Equal Justice Initiative (EJI) "Lynching in America" database and Monroe Work's "Negro Year Book" historical record.
- **Resolution**: County | **Temporal**: 1877-1950 historical
- **URL**: EJI: https://eji.org/research/lynching-in-america/ (county data downloadable via their report); Monroe Work Today project
- **Why it matters**: Lynching geography predicts modern racial attitudes and racial wealth gaps (Cook, Logan, & Parman 2014). Counties with higher historical lynching counts have lower current Black voter registration rates and more racially polarized politics. The mechanism is both direct demographic (Black out-migration from terror) and attitudinal (persistent white racial dominance). FL, GA, AL had among the highest lynching rates nationally.
- **Signal**: Racial violence history as community structural legacy. EJI specifically found FL and AL had the highest per-capita lynching rates in the nation — this maps directly onto current racial political polarization geography.
- **Cost/Effort**: Free, EJI downloadable county data. Low integration effort; single data table.

---

### Immigration & Demographics

### 81. ICE Arrests and Deportations by County (TRAC / ERO)
- **What**: County-level ICE enforcement arrest and removal (deportation) counts. TRAC Syracuse aggregates ICE Enforcement and Removal Operations (ERO) data via FOIA requests.
- **Resolution**: County of arrest | **Temporal**: 2003-present
- **URL**: https://tracreports.org/ (TRAC Immigration; subscription for detailed data, but public summaries free); FOIA releases via MuckRock
- **Why it matters**: ICE enforcement intensity is politically salient and varies dramatically by jurisdiction. Sanctuary policies (Miami-Dade reversed its sanctuary status in 2017) vs. 287(g) cooperation vary by county sheriff. Counties with high ICE activity have politically mobilized immigrant communities AND potentially politically mobilized nativist communities simultaneously.
- **Signal**: Immigration enforcement as community political flash point. FL counties with 287(g) sheriff agreements show distinct political trajectories from sanctuary counties.
- **Cost/Effort**: Free public summaries from TRAC; detailed county data requires subscription (~$20/month academic rate). MuckRock FOIA aggregations available.

### 82. DACA Recipient Concentration (USCIS Data)
- **What**: State-level and some county-level estimates of DACA recipients — their concentration, age, employment sector.
- **Resolution**: State (top-line), county (estimated from American Immigration Council research)
- **URL**: USCIS DACA data: https://www.uscis.gov/DACA | American Immigration Council county-level analysis reports
- **Why it matters**: DACA recipients are concentrated in specific counties (Miami-Dade, Broward, Gwinnett GA, Jefferson AL). Their concentration creates both political constituency pressure (their families vote) and political backlash signals. The uncertainty of their status (DACA program has been repeatedly litigated) creates persistent political salience.
- **Signal**: Immigration community presence and political stakes. High DACA concentration counties in GA (Gwinnett, Cobb) track closely with the suburban Atlanta Hispanic political realignment.
- **Cost/Effort**: Free, USCIS quarterly snapshots. County-level requires academic report compilation.

### 83. H-2A Agricultural Guest Worker Program Data
- **What**: County-level counts of H-2A temporary agricultural visa certifications — workers requested, positions certified, employer names, crop types.
- **Resolution**: County (employer address) | **Temporal**: Annual
- **URL**: DOL OFLC Performance Data: https://www.dol.gov/agencies/eta/foreign-labor/performance (downloadable Excel)
- **Why it matters**: H-2A workers are overwhelmingly from Mexico and Central America, concentrated in rural agricultural counties. FL leads all states in H-2A certifications (~30% of national total) — concentrated in Immokalee (Collier), Homestead (Miami-Dade), and the Glades (Belle Glade/Palm Beach). These are majority-Hispanic agricultural communities with distinct political dynamics. GA's vidalia onion and poultry counties also show high H-2A use.
- **Signal**: Agricultural labor immigration concentration. H-2A counties are distinctive — large temporary immigrant populations, abusive labor conditions, low voter turnout despite large Hispanic population. Captures the "invisible" agricultural labor geography.
- **Cost/Effort**: Free, DOL Excel download. County-level from employer address. Trivial to integrate.

### 84. Refugee Resettlement by County (State Dept. / WRAPS)
- **What**: County-level refugee resettlement placements by year, country of origin, and sponsoring agency.
- **Resolution**: County | **Temporal**: 1975-present
- **URL**: State Dept. Refugee Processing Center (RP): https://www.wrapsnet.org/admissions-and-arrivals/ | Wilson Sheehan Lab for Economic Opportunities (Notre Dame) maintains academic dataset
- **Why it matters**: Refugee resettlement is highly concentrated — most resettlees go to specific urban counties with resettlement agency infrastructure. Atlanta (Clarkston GA is called "the most diverse square mile in America"), Jacksonville FL, and Birmingham AL have significant refugee communities. Resettlement creates both multicultural community building AND (in some areas) political backlash among native-born residents.
- **Signal**: Rapid ethnic community formation as political stress signal. Clarkston GA is a documented case study of community transformation and political tension; replicated in smaller form across the South.
- **Cost/Effort**: Free, State Dept. data download. County-level, annual, CSV format.

---

### Health & Social Signals

### 85. CDC WONDER Natality Data — Birth Rates and Teen Birth Rates
- **What**: County-level birth rates, teen birth rates (15-19), unmarried birth rates, maternal age distribution, prenatal care timing.
- **Resolution**: County | **Temporal**: 1995-2024
- **URL**: https://wonder.cdc.gov/natality.html (free, API accessible)
- **Why it matters**: Teen birth rates are among the strongest single-variable predictors of educational attainment and economic mobility at the county level. High teen birth rates signal communities with limited economic opportunity for women and social conservatism. The Deep South (rural AL, rural GA) has persistently high teen birth rates — a community health signature that predates and predicts political outcomes.
- **Signal**: Community economic opportunity for women as reflected in reproductive behavior. Teen birth rate predicts poverty persistence better than current income, and predicts Republican lean even controlling for income.
- **Cost/Effort**: Free, CDC WONDER API. County-level, clean data.

### 86. Eviction Lab Data — Eviction Rates (Princeton)
- **What**: County-level eviction filing rates and judgment rates (2000-2018), and current eviction tracking from the Eviction Tracking System (ETS, 31 cities).
- **Resolution**: County (historical) / ZIP (ETS) | **Temporal**: 2000-2018 (historical), 2020-present (ETS)
- **URL**: https://evictionlab.org/get-the-data/ (email registration required for download)
- **Why it matters**: Eviction rate is a measure of housing precarity and landlord power that poverty rates miss. High eviction counties have transient populations with low voter registration and high political disengagement. Matthew Desmond's "Evicted" documented how eviction perpetuates poverty in Milwaukee — the same dynamic applies to Atlanta, Jacksonville, Birmingham.
- **Signal**: Housing precarity and political disengagement. High eviction counties may show low Democratic base turnout even in communities with favorable demographics — eviction disrupts voter registration.
- **Cost/Effort**: Free with email registration. County-level CSV. Princeton Eviction Lab is research-grade data.

### 87. Maternal Mortality Rates by County (CDC / AHRQ)
- **What**: County-level maternal mortality rates (deaths per 100,000 live births), racial disparities in maternal mortality, hospital delivery data.
- **Resolution**: County (some suppression in small counties) | **Temporal**: 2016-present
- **URL**: CDC WONDER maternal mortality tables; AHRQ Healthcare Cost and Utilization Project (HCUP); March of Dimes Maternity Care Deserts report
- **Why it matters**: US maternal mortality is among the highest in developed nations and is racially stratified — Black mothers die at 2-3x the rate of white mothers. The "maternity care desert" (counties with no hospital offering obstetric care) is concentrated in rural AL, GA, MS. This is a policy salience issue with specific political implications for healthcare access and reproductive rights politics.
- **Signal**: Healthcare system failure as community political grievance. March of Dimes found 36% of US counties are maternity care deserts — concentrated in the South, directly in the FL/GA/AL study area.
- **Cost/Effort**: Free, CDC WONDER. Some county suppression for small populations. March of Dimes publishes county-level CSV.

### 88. Disability Rate and SSDI Recipiency (SSA)
- **What**: County-level Social Security Disability Insurance (SSDI) and Supplemental Security Income (SSI) recipient rates, disability type distribution.
- **Resolution**: County | **Temporal**: Annual
- **URL**: SSA Annual Statistical Report on SSDI: https://www.ssa.gov/policy/docs/statcomps/di_asr/ | SSA geographic data download
- **Why it matters**: High disability rates are both a health indicator and an economic dependency signal. Rural Appalachian and Deep South counties have disability rates 2-3x the national average — partly reflecting genuine occupational injury (mining, agriculture) and partly reflecting disability as the "last resort" benefit after UI exhaustion. Disability dependency predicts political attitudes toward government benefit programs.
- **Signal**: Long-term economic dependency and working-age health failure. High SSDI counties in AL/GA rural areas represent the intersection of deindustrialization, poor health infrastructure, and intergenerational poverty.
- **Cost/Effort**: Free, SSA downloads. County-level, annual. Clean data.

### 89. Alcohol Outlet Density (TTB / State Licensing)
- **What**: Number of licensed alcohol retailers, bars, and restaurants serving alcohol per county, from state alcohol beverage control (ABC) licensing data and federal TTB.
- **Resolution**: County | **Temporal**: Annual
- **URL**: FL Div. of Alcoholic Beverages & Tobacco; GA DOR; AL ABC Board — all publish licensee databases; also via SafeGraph POI data
- **Why it matters**: Alcohol outlet density correlates with both social disorder (crime, DUI deaths) and community type (bar culture vs. dry counties). Alabama still has dry counties (nearly 1/3 as of 2020) — a direct measure of cultural conservatism. FL's liquor license density in tourist areas vs. rural areas is a stark community type differentiator.
- **Signal**: Cultural conservatism signal (dry vs. wet county) and social disorder proxy. Dry county status is the most direct available measure of traditional Baptist/evangelical cultural governance.
- **Cost/Effort**: Free, state agency databases. Some compilation required; SafeGraph POI historical data is the cleanest aggregation.

---

### Technology & Digital Signals

### 90. Starlink Satellite Internet Adoption (SpaceX / FCC)
- **What**: FCC broadband availability data for Starlink coverage + SpaceX's own coverage maps; state-level subscriber counts where available; academic estimates of rural Starlink adoption from satellite imagery and FCC filings.
- **Resolution**: County (estimated from FCC data) | **Temporal**: 2021-present
- **URL**: FCC National Broadband Map (includes Starlink as fixed wireless); SpaceX SEC filings for subscriber counts; Pew Research rural internet studies
- **Why it matters**: Starlink specifically targets rural counties with no wired broadband — exactly the communities in this study. High Starlink adoption signals rural tech adoption AND willingness to pay premium prices for connectivity. Starlink users may be a distinct socioeconomic subgroup within rural communities (higher income, more connected despite rural location). The Starlink → rural political information environment change is understudied.
- **Signal**: Rural digital connectivity frontier. Starlink adoption distinguishes connected rural communities from fully isolated ones — may predict shifts in information environment and political exposure.
- **Cost/Effort**: Moderate. FCC broadband map includes Starlink provider data. Academic proxies available.

### 91. Solar Panel Installation Rates (Lawrence Berkeley Lab / SEIA)
- **What**: County-level residential and commercial solar photovoltaic installation counts, capacity (kW), and cost trends.
- **Resolution**: County | **Temporal**: 2000-present
- **URL**: Lawrence Berkeley National Lab Tracking the Sun: https://emp.lbl.gov/tracking-the-sun (annual county data, free)
- **Why it matters**: Solar adoption is both an economic signal (payback period economics work best in sunny, high-electricity-cost areas) and a political identity signal. FL is a paradox — one of the sunniest states with historically low residential solar adoption due to utility lobbying. Counties where rooftop solar has penetrated despite utility opposition show distinct political cultures. Solar map mirrors college-education map in FL.
- **Signal**: Clean energy adoption as community political culture marker. Coastal FL + Orlando suburbs have high solar; Panhandle has near-zero. This perfectly tracks the political gradient.
- **Cost/Effort**: Free, LBNL annual release. County-level CSV. Very clean data.

### 92. FCC Spectrum Auction Winners / Wireless Coverage by County
- **What**: Mobile wireless network coverage quality by county — 4G LTE and 5G coverage percentage, number of competing carriers, spectrum holdings.
- **Resolution**: Census block / county | **Temporal**: Annual (FCC semi-annual)
- **URL**: FCC National Broadband Map (wireless layer): https://broadbandmap.fcc.gov/data-download
- **Why it matters**: Mobile-only internet users are concentrated in low-income rural communities where no wired broadband exists. Counties relying on 3G or spotty 4G are information-isolated in ways the FCC broadband availability map (which counts wireline) misses. In rural AL and GA, Verizon/AT&T rural coverage has major gaps — these communities may have fundamentally different information access patterns.
- **Signal**: Mobile internet quality as information environment proxy. Rural coverage quality may predict political information sources (radio > streaming) and engagement patterns.
- **Cost/Effort**: Free, FCC bulk data. Block-level, needs county aggregation.

---

### Protest, Civil Conflict & Social Movements

### 93. ACLED US Protest and Political Violence Database
- **What**: Event-level database of political demonstrations, riots, and political violence in the US, geocoded to city/county. Covers BLM protests (2020), January 6 precursor events, anti-vaccine rallies, etc.
- **Resolution**: Event point (aggregatable to county) | **Temporal**: 2020-present (US coverage)
- **URL**: https://acleddata.com/ (free for researchers with registration; academic license)
- **Why it matters**: Protest activity is a direct measure of political mobilization and community conflict. BLM protest density in 2020 maps onto communities that subsequently showed Democratic shifts. Anti-mask rally density maps onto communities that shifted further Republican. The spatial distribution of political activism is a leading indicator that precedes vote share changes.
- **Signal**: Political mobilization and conflict as community signal. Counties with high BLM protest activity in 2020 showed different 2020 shifts than demographically similar counties without protests.
- **Cost/Effort**: Free with researcher registration. CSV exports by country/date range. Good for 2020-present; limited historical US coverage.

### 94. Washington Post Police Shooting + Fatal Force Database (State-Level Policy Variation)
- **What**: Beyond individual incidents, aggregate state and county policy variables: use-of-force policies, duty to intervene, de-escalation training requirements, civilian review boards, police union contract strength.
- **Resolution**: Department/county | **Temporal**: Current
- **URL**: Campaign Zero Policy Scorecard: https://campaignzero.org/policyplatform/ | Invisible Institute Accountability Data: https://invisible.institute/ | NACOLE (National Association for Civilian Oversight of Law Enforcement)
- **Why it matters**: Police accountability policy structure varies dramatically across counties and correlates with political culture. Counties with strong police unions and no civilian oversight have different political dynamics from reform-minded jurisdictions. The 2020 "defund" debate polarized communities along lines that correlated with existing political geography.
- **Signal**: Law enforcement accountability culture as community political marker. Strong police union counties show specific Republican-leaning patterns among pro-police voters; reform jurisdictions show Democratic-leaning patterns.
- **Cost/Effort**: Free, Campaign Zero and Invisible Institute. City/department level; requires county-level aggregation.

---

### Government & Fiscal Signals

### 95. Census of Governments — Local Tax Revenue and Expenditure
- **What**: Every unit of local government (county, city, special district, school district) with revenue sources (property tax, sales tax, fees), expenditure categories, debt, and employment.
- **Resolution**: Government unit (aggregatable to county) | **Temporal**: Every 5 years (2012, 2017, 2022) + annual for larger governments
- **URL**: https://www.census.gov/programs-surveys/cog.html (Census of Governments)
- **Why it matters**: Local government fiscal structure reveals political philosophy in action. Counties with high property tax revenue fund services differently than those relying on sales taxes. Government employment as a share of county employment captures public sector dependency. Debt per capita signals infrastructure investment vs. fiscal conservatism.
- **Signal**: Revealed fiscal political philosophy. High per-pupil school spending = revealed preference for education investment. Low property tax, high sales tax = regressive fiscal structure. These reveal governing political culture beyond election outcomes.
- **Cost/Effort**: Free, Census download. Quinquennial; large files. Some aggregation from government unit to county.

### 96. School Board Election Results and Culture War Indicators
- **What**: Precinct/county-level school board election results, plus tracking of school board recall attempts, book ban policies, critical race theory (CRT) bans, sex ed curriculum controversies.
- **Resolution**: District | **Temporal**: 2018-present (culture war era)
- **URL**: OpenElections (for election results) + Banned Books Week ALA database + PEN America Index of School Book Bans + FL Dept. of Education curriculum objections log
- **Why it matters**: School board elections became the primary battleground for culture war politics after 2020. Florida has the most documented book banning activity (Duval, Clay, Brevard counties). School board election results predict the trajectory of culture war intensity — they are leading indicators of the community's future political direction.
- **Signal**: Culture war intensity and trajectory. A county where the culture war slate won the school board in 2022 is likely to show increased Republican shift in 2024 — and vice versa for reform slate victories.
- **Cost/Effort**: Free, but requires multi-source compilation. PEN America and ALA publish ban databases. FL DoE has its own public log of curriculum objections.

### 97. Small Area Income and Poverty Estimates (SAIPE) — Annual County Poverty Rates
- **What**: Annual county-level poverty rate and median household income estimates, updated every year using tax data and survey methods. More current than ACS 5-year estimates.
- **Resolution**: County | **Temporal**: Annual, 1989-present
- **URL**: https://www.census.gov/programs-surveys/saipe.html (Census Bureau)
- **Why it matters**: Annual updates make SAIPE more temporally sensitive than ACS 5-year estimates — can track recession impacts (2008-2010 spike), pandemic effects (2020 dip due to stimulus), and recovery trajectory. Year-over-year poverty change may predict political mood better than absolute poverty level.
- **Signal**: Economic welfare trajectory at high temporal resolution. Poverty rate change between election cycles may predict shift direction better than the level.
- **Cost/Effort**: Free, Census API. Annual county-level. Trivial to integrate.

---

### Social Mobility & Opportunity

### 98. Chetty/Raj Opportunity Atlas — Upward Mobility by County (Race/Income)
- **What**: County-level estimates of upward economic mobility for children raised in the bottom quintile, broken down by race (Black, white, Hispanic) and parent income. Based on 20M+ IRS tax records.
- **Resolution**: County/census tract | **Temporal**: Birth cohorts 1978-1983 (now adults)
- **URL**: https://opportunityatlas.org/ (downloadable CSV) | https://opportunityinsights.org/data/
- **Why it matters**: This is distinct from the Opportunity Insights Social Capital data already cataloged. The Opportunity Atlas specifically measures whether poor children escape poverty — and crucially does so by race. A county with high Black child mobility but low white child mobility is structurally different from a county where both are low. This racial mobility gap is a direct measure of structural racism's economic effect.
- **Signal**: Intergenerational economic mobility as community defining feature. Low-mobility counties (especially for Black children) in GA/AL predict political disengagement among Black voters and political frustration among white voters simultaneously.
- **Cost/Effort**: Free, direct CSV download. County and tract level. Clean data.

### 99. Social Capital Project — Congressional Subcommittee County Data
- **What**: JD Vance-era Senate subcommittee data on social capital, civic engagement, volunteering, associational membership, and trust by county. Based on multiple survey sources.
- **Resolution**: County | **Temporal**: ~2018 snapshot
- **URL**: https://www.jec.senate.gov/public/index.cfm/republicans/2018/4/the-geography-of-social-capital-in-america (Joint Economic Committee, Republican staff)
- **Why it matters**: This is a right-of-center academic project that produced county-level social capital estimates — covering civic participation, family stability, community cohesion, and institutional trust. It's conceptually distinct from Chetty's mobility data: it measures community-level social infrastructure rather than individual economic outcomes.
- **Signal**: Community cohesion and institutional trust as political moderators. Low social capital counties may show more populist/anti-institutional political behavior.
- **Cost/Effort**: Free, JEC website. County-level, single download.

---

### Deep Historical

### 100. Historical Migration Data — Great Migration Destination Counties
- **What**: County-level data on Black population change during the Great Migration (1910-1970) — origin counties in the South and destination counties in the North/West, with net migration estimates.
- **Resolution**: County | **Temporal**: 1910-1970 (decennial census)
- **URL**: IPUMS NHGIS historical census tables; Boustan (2017) "Competition in the Promised Land" replication data (Princeton)
- **Why it matters**: The Great Migration depopulated rural Deep South Black communities while creating the Northern Black political base. Counties that lost large Black populations in the Great Migration show persistent structural effects — less Black political power, different racial composition of the political coalition. Leah Boustan's work shows specific Great Migration flow data at county level.
- **Signal**: Demographic path dependency. Which southern counties "sent" Black migrants to Chicago/Detroit vs. which retained Black populations shapes current racial political geography. FL (Miami's Overtown, Jacksonville) vs. Black Belt AL/GA shows this divergence clearly.
- **Cost/Effort**: Free, NHGIS. Historical census race tables by county. Requires crosswalking to current boundaries.

### 101. Reconstruction-Era Black Officeholding (Freedom on the Move / UNC)
- **What**: County-level records of Black elected officials during Reconstruction (1865-1876) and during the Second Reconstruction (1965-2000).
- **Resolution**: County | **Temporal**: 1865-1876 (Reconstruction), 1965-2000 (post-VRA)
- **URL**: "Freedom on the Move" (Cornell, Yale, UNC) escaped slave advertisement database adjacent; Foner's "Freedom's Lawmakers" replication data; Joint Center for Political and Economic Studies Black elected officials series
- **Why it matters**: Counties with strong Reconstruction-era Black political representation were systematically destroyed during Redemption (1876-1900). The geographic pattern of Redemption violence and Black political erasure predicts modern racial political polarization. Counties where Black political power was never restored after Redemption have distinct political structures.
- **Signal**: Democratic institutional legacy of racial political exclusion. The contrast between counties that had Black sheriffs/legislators during Reconstruction vs. those that didn't predicts modern Black political mobilization patterns.
- **Cost/Effort**: Moderate. Academic datasets; some manual compilation from historical records.

---

*End of Expansion Round 2. Total ideated sources: 101 (40 original + 61 new).*

---

## Expansion Round 3 — GitHub Repositories & Tools

*Added 2026-03-19. Per Hayden's directive: "GitHub has a myriad of projects that may have data sources or chunks of tools we could use."*

These are GitHub repos that either contain pre-compiled county-level data we can ingest directly, or tools that simplify fetching data we've already identified.

---

### Pre-Compiled County-Level Datasets

### 102. CountyPlus — Open-Source County Panel Dataset
- **What**: 3,000+ US counties × years 2003-2019. Pre-compiled panel with economic, demographic, health, and social variables from all major public sources (BEA, BLS, Census, CDC, USDA).
- **Resolution**: County | **Temporal**: 2003-2019
- **URL**: https://github.com/Clpr/CountyPlus
- **Why it matters**: Single-download dataset that consolidates many of the individual sources we've listed (BEA income, BLS unemployment, USDA rural codes, CDC health). Could bootstrap feature engineering for 10+ Tier 1/2 sources with one integration. Panel format means temporal change features are pre-aligned.
- **Signal**: Meta-source — accelerates integration of multiple individual sources.
- **Cost/Effort**: Free. Single CSV/parquet download. May need updating beyond 2019.

### 103. JsonOfCounties — Multi-Source County Data in JSON
- **What**: County-level data including BEA income, presidential election results, demographics, education, and health metrics compiled into JSON format.
- **Resolution**: County | **Temporal**: Various
- **URL**: https://github.com/evangambit/JsonOfCounties
- **Why it matters**: Pre-joined dataset covering income, elections, education, health — ready to cross-reference against our shift vectors. JSON format may need conversion but the joining work is done.
- **Signal**: Convenience source — validates our own feature builds and fills gaps.
- **Cost/Effort**: Free. JSON files, trivial conversion.

### 104. US County Level Election Results 2008-2024
- **What**: County-level presidential results for 2008, 2012, 2016, 2020, 2024 from NYT, Guardian, Politico, Fox News. Standardized CSV format.
- **Resolution**: County | **Temporal**: 2008-2024
- **URL**: https://github.com/tonmcg/US_County_Level_Election_Results_08-24
- **Why it matters**: Pre-compiled 5-cycle presidential results at county level. Could supplement VEST data for quick shift vector computation or validation. Already cleaned and standardized — no scraping needed.
- **Signal**: Validation source for our VEST/MEDSL election returns. Also extends to 2008/2012 which we don't currently use.
- **Cost/Effort**: Free. Single CSV per cycle.

### NYTimes Open Source Election Maps

### 105. NYT 2024 Presidential Precinct Map (NEW — not in prior list)
- **What**: Precinct-level TopoJSON/GeoJSON results for the 2024 presidential election. Published by NYT as open data. Includes vote totals, margins, and precinct boundaries.
- **Resolution**: Precinct | **Temporal**: 2024
- **URL**: https://github.com/nytimes/presidential-precinct-map-2024
- **Why it matters**: This is the 2024 companion to the 2020 precinct data already listed (#41). Having both years at precinct level enables sub-county shift vector computation — a major upgrade from county-level shifts. FL and GA have substantial coverage.
- **Signal**: Sub-county shift vectors. The highest-resolution temporal shift data publicly available.
- **Cost/Effort**: Free, direct GeoJSON download. Requires spatial join to tracts.

### 106. NYT Upshot 2020 Presidential Precinct Map
- **What**: GeoJSON with Biden/Trump vote totals per precinct, plus generated precinct boundaries from L2 voter file.
- **Resolution**: Precinct | **Temporal**: 2020
- **URL**: https://github.com/TheUpshot/presidential-precinct-map-2020
- **Why it matters**: Already listed as #41 but noting the actual GitHub repo for direct download. GeoJSON format with GEOID = county FIPS + precinct ID.
- **Signal**: See #41.
- **Cost/Effort**: Free.

### 107. Election Geodata — Precinct Shapes Archive
- **What**: Comprehensive archive of precinct boundary shapefiles across multiple election cycles. Community-maintained with contributions from election officials and academics.
- **Resolution**: Precinct | **Temporal**: Multiple cycles
- **URL**: https://github.com/nvkelso/election-geodata
- **Why it matters**: The boundary files needed to geolocate precinct results from OpenElections or MEDSL. Without boundaries, precinct results can't be mapped to tracts.
- **Signal**: Infrastructure — enables sub-county analysis from multiple data sources.
- **Cost/Effort**: Free. Shapefile format, varies by state/year.

### Health & Opioid Crisis Data

### 108. OEPS — Opioid Environment Policy Scan (GeoDaCenter/UChicago)
- **What**: Multi-scale data warehouse (tract, zip, county, state) covering opioid prescribing rates, naloxone access, treatment facility locations, MAT provider counts, plus socioeconomic context (ACS, health, employment). Pre-built for opioid research.
- **Resolution**: Census tract to state | **Temporal**: Various (2015-2022)
- **URL**: https://github.com/GeoDaCenter/opioid-policy-scan
- **Why it matters**: The most comprehensive pre-compiled opioid crisis dataset at tract level. Includes treatment access (buprenorphine providers, MAT facilities), prescribing rates, and naloxone pharmacy availability — all at sub-county resolution. These are direct measures of the "deaths of despair" crisis that our CDC WONDER mortality fetcher captures from the outcome side.
- **Signal**: Opioid crisis input variables (treatment access, prescribing) complement mortality output variables. Counties with high mortality but low treatment access are a distinct community type from those with high mortality and high treatment access.
- **Cost/Effort**: Free. Pre-built CSV files organized by spatial scale. Very low effort.

### 109. US Overdoses Analysis Database
- **What**: County-level opioid mortality 1999-2015 merged with ACS income, medical insurance, SAMHSA drug use surveys, naloxone/buprenorphine dispensing data.
- **Resolution**: County/state | **Temporal**: 1999-2015
- **URL**: https://github.com/mattkiefer/us-overdoses
- **Why it matters**: Pre-joined dataset linking overdose mortality to economic and healthcare access variables. The temporal range (1999-2015) captures the pre-Trump opioid crisis that shaped the 2016 political realignment.
- **Signal**: Historical opioid crisis trajectory — counties where the crisis peaked before 2016 may have shifted differently from those where it peaked later.
- **Cost/Effort**: Free. SQLite + CSV format.

### Political Science Research Data

### 110. PolData — Master Index of Political Datasets
- **What**: Curated catalog of 300+ political science datasets across 15 categories: elections, parties, institutions, policy, public opinion, media, conflict, etc. Each entry includes topic, coverage, URL, and format.
- **Resolution**: Varies | **Temporal**: Varies
- **URL**: https://github.com/erikgahner/PolData
- **Why it matters**: This is a meta-source — a research-grade index of political datasets that may contain sources we haven't identified. Worth scanning the full list for FL/GA/AL-relevant county-level data. Maintained by a political scientist.
- **Signal**: Discovery accelerator — may identify 5-10 additional sources relevant to our specific geographic focus.
- **Cost/Effort**: Free. Markdown index file.

### 111. US Polling Places Database (Center for Public Integrity)
- **What**: Standardized geocoded polling place locations for general elections across multiple states and years. Includes address, precinct assignment, and temporal changes.
- **Resolution**: Point (polling place) | **Temporal**: 2012-2020
- **URL**: https://github.com/PublicI/us-polling-places
- **Why it matters**: Polling place closures and relocations are a documented voter suppression mechanism. Counties that reduced polling places between elections show lower turnout, particularly in minority communities. GA's 2018 election featured specific polling place closure controversies (Randolph County).
- **Signal**: Election access infrastructure as political participation signal. Polling place distance increase predicts turnout decline.
- **Cost/Effort**: Free. CSV format, geocoded.

### Tools for Data Fetching

### 112. censusdis — Pythonic Census API Wrapper
- **What**: Python package for programmatic access to all 1,500+ Census Bureau API endpoints. Handles geography hierarchies, variable discovery, and data download with pandas integration.
- **Resolution**: All Census geographies (tract to national) | **Temporal**: All Census/ACS years
- **URL**: https://github.com/censusdis/censusdis
- **Why it matters**: Could replace our manual ACS fetching with a more robust, discoverable API wrapper. Supports geographic boundary downloads alongside data — useful for spatial joins. More Pythonic than the `census` package.
- **Signal**: Tool — accelerates Census/ACS data integration and makes it easier to explore new Census tables.
- **Cost/Effort**: Free. `pip install censusdis`. Could refactor our ACS pipeline to use this.

### 113. censusapi — R Census API Wrapper
- **What**: Lightweight R package for all Census Bureau APIs. Supports data discovery, all geographies, and all endpoints including SAIPE, BDS, CBP, ACS, Decennial.
- **Resolution**: All Census geographies | **Temporal**: All available years
- **URL**: https://github.com/hrecht/censusapi
- **Why it matters**: For our R pipeline (MRP/propagation), this provides native R access to Census data without Python intermediary. Particularly useful for SAIPE annual poverty data (source #97) and BDS business dynamics (source #17).
- **Signal**: Tool — R-native Census access for the propagation pipeline.
- **Cost/Effort**: Free. `install.packages("censusapi")`.

---

### Priority GitHub Sources to Integrate First

| Source | # | Why First | Effort |
|--------|---|-----------|--------|
| CountyPlus panel | 102 | Bootstraps 10+ features at once | Very Low |
| NYT 2024 precinct map | 105 | Sub-county shift vectors | Medium |
| OEPS opioid scan | 108 | Tract-level crisis data, pre-built | Very Low |
| PolData index | 110 | May reveal new sources | Research only |
| censusdis tool | 112 | Improves all Census fetching | Medium (refactor) |

---

*End of Expansion Round 3. Total ideated sources: 113 (101 prior + 12 new from GitHub survey).*

---

## Expansion Round 4 — Creative & Unconventional Sources

*Added 2026-03-19. Per Hayden's directive: be aggressive, creative, think turnout, demo, marketing, politics, whatever. Exploring underrepresented signal categories: partisan media, extremism, consumer culture, financial predation, family structure, civic institutions, tourism economy, student debt, and more.*

---

### Partisan Media & Information Environment

### 114. Fox News / MSNBC Cable Ratings by DMA (Nielsen)
- **What**: Cable news viewership ratings (Fox News, MSNBC, CNN) by Designated Market Area (~210 DMAs). Primetime viewership shares and total day viewing.
- **Resolution**: DMA (mappable to county clusters) | **Temporal**: Monthly/quarterly, 2000-present
- **URL**: Nielsen ratings published via trade press (Adweek, TVNewser); academic access via Nielsen Media Research; some DMA-level data via cable operator FCC filings
- **Why it matters**: Cable news viewership is the strongest revealed-preference measure of partisan information diet. The ratio of Fox:MSNBC viewership in a DMA directly predicts partisan lean. More importantly, CHANGE in this ratio over time captures information environment shifts that precede vote shifts. The "Fox News effect" (DellaVigna & Kaplan 2007) showed measurable Republican vote share increase in markets where Fox launched.
- **Signal**: Information diet as community political identity. Fox-dominant DMAs may show uniform national-wave shifts; mixed DMAs may show more heterogeneous responses.
- **Cost/Effort**: Mixed — trade press publishes partial DMA ratings free; full data requires Nielsen subscription or academic access.

### 115. Talk Radio Station Density & Format (iHeartMedia / Cumulus)
- **What**: AM/FM talk radio station formats by county — conservative talk, progressive talk, sports talk, religious, Spanish-language — with signal coverage maps. Extends source #37 (radio formats) with specific partisan talk radio focus.
- **Resolution**: Station signal coverage area | **Temporal**: FCC license database (current), historical via archived databases
- **URL**: FCC CDBS license database: https://www.fcc.gov/media/radio/cdbs-data-files | Radio-Locator.com (format tracking)
- **Why it matters**: Conservative talk radio (Limbaugh legacy, Hannity, Shapiro, local hosts) has far greater geographic reach than cable news in rural areas. Many rural FL/GA/AL counties have 3+ conservative AM talk stations and zero progressive stations. This creates an asymmetric information environment that cable ratings alone don't capture. The decline of local AM radio leaves national syndicated talk as the primary political voice.
- **Signal**: Asymmetric partisan information exposure in rural areas. Counties with only conservative talk radio have a fundamentally different political media diet than metro areas with diverse options.
- **Cost/Effort**: Free (FCC database). Requires format classification and signal coverage mapping.

### 116. Podcast Download Geography (Chartable / Spotify for Podcasters)
- **What**: Geographic distribution of political podcast listenership — top political podcasts by DMA or zip code, from Spotify for Podcasters (formerly Anchor) analytics and Chartable rankings.
- **Resolution**: DMA/ZIP | **Temporal**: 2018-present
- **URL**: Chartable (https://chartable.com/) publishes geographic popularity rankings; Edison Research "Infinite Dial" survey includes podcast geography; Spotify for Podcasters (limited public data)
- **Why it matters**: Podcast consumption has partly replaced talk radio for younger audiences. Political podcast geography differs from cable news — Joe Rogan, Ben Shapiro, Pod Save America have distinct geographic listener profiles that don't perfectly align with cable news DMA patterns. This captures under-45 political media consumption.
- **Signal**: Next-generation partisan information environment. Podcast geography may predict political shifts in younger cohorts before they appear in election results.
- **Cost/Effort**: Limited free data; Edison Research publishes annual surveys; Chartable rankings are public.

### 117. Local Facebook Group Political Activity (CrowdTangle Archive)
- **What**: Archived CrowdTangle data (Meta's now-shuttered research tool) on local political Facebook group membership, posting frequency, and engagement patterns by geographic area.
- **Resolution**: County (estimated from group location) | **Temporal**: 2017-2023 (CrowdTangle sunset)
- **URL**: Social Science One archived datasets; Meta Content Library (successor to CrowdTangle, academic access)
- **Why it matters**: Local political Facebook groups ("Patriots of [County]", "[City] Moms for Liberty", etc.) are a direct measure of grassroots political organizing. Groups that gained explosive membership 2020-2022 may predict which counties are experiencing political mobilization surges.
- **Signal**: Grassroots political organizing intensity. Counties with high-engagement local political groups may show more extreme shifts than counties where political activity stays passive.
- **Cost/Effort**: Moderate — archived CrowdTangle data available via academic partnerships. Meta Content Library requires institutional application.

---

### Extremism & Hate Groups

### 118. SPLC Hate Group Map — Hate Group Density by County
- **What**: Southern Poverty Law Center's annual tracking of active hate groups (KKK, neo-Nazi, white nationalist, anti-LGBTQ, anti-Muslim, Black separatist) by location and type.
- **Resolution**: City (mappable to county) | **Temporal**: 1999-present, annual
- **URL**: https://www.splcenter.org/hate-map (interactive, downloadable)
- **Why it matters**: Hate group density per capita is a direct measure of organized extremist presence. FL, GA, and AL consistently rank in the top 10 states for hate group count. The relationship between hate group presence and political behavior is complex — it signals both community grievance and political mobilization capacity for extremist candidates.
- **Signal**: Organized extremism as community political marker. Counties with active hate groups may show distinct intra-Republican primary dynamics and amplified cultural backlash responses to demographic change.
- **Cost/Effort**: Free, SPLC downloadable dataset. City-level, needs county aggregation.

### 119. Confederate Monument Density (SPLC / Whose Heritage?)
- **What**: Geocoded locations of Confederate monuments, statues, school names, road names, and other public symbols across the South. Includes installation date, removal status, and type.
- **Resolution**: Point (aggregatable to county) | **Temporal**: Installation dates 1860s-present; SPLC tracks removals since 2015
- **URL**: https://www.splcenter.org/whose-heritage (downloadable dataset with latitude/longitude)
- **Why it matters**: Confederate monument density is a proxy for historical Lost Cause commemoration culture — most monuments were erected 1900-1920 during Jim Crow, not after the Civil War. Counties that installed monuments later (1950s-1960s, during Civil Rights era) signal different community attitudes than those with Reconstruction-era memorials. Monument removal battles (2017-present) are a direct measure of culture war intensity.
- **Signal**: Confederate memory as living political identity. Counties fighting to preserve monuments in 2020s are experiencing active cultural political conflict. AL passed a state law prohibiting monument removal (2017 Memorial Preservation Act) — the legal fight itself is a political signal.
- **Cost/Effort**: Free, SPLC dataset. Point data with coordinates. Very low effort.

### 120. Anti-Defamation League (ADL) Hate Incident Reports
- **What**: County-level hate incidents, hate crimes, and extremist events tracked by ADL from news reports, law enforcement data, and community reports.
- **Resolution**: City/county | **Temporal**: 2002-present
- **URL**: https://www.adl.org/heat-map (interactive, data downloads available for researchers)
- **Why it matters**: Hate incidents (distinct from SPLC hate groups — incidents are events, not organizations) capture the behavioral expression of extremism. Spikes in hate incidents correlate with political events (post-2016 election, post-Charlottesville, post-January 6). Geographic clustering of incidents reveals where political tension is highest.
- **Signal**: Political tension and racial conflict intensity. Counties with spiking hate incidents may be communities in rapid demographic or political transition.
- **Cost/Effort**: Free with researcher registration.

---

### Consumer & Retail Culture Geography

### 121. Cracker Barrel vs. Whole Foods Index
- **What**: Relative density of "red" retail brands (Cracker Barrel, Bass Pro Shops, Tractor Supply, Hobby Lobby) vs. "blue" retail brands (Whole Foods, Trader Joe's, REI, Lululemon) by county.
- **Resolution**: County (from point locations) | **Temporal**: Current snapshot
- **URL**: Company store locators + OpenStreetMap POI data + SafeGraph/Dewey (historical patterns)
- **Why it matters**: Dave Wasserman's famous "Cracker Barrel vs. Whole Foods" index (2012) showed that the presence of these stores predicted county-level presidential vote better than income or education. The principle extends to a broader consumer landscape — Tractor Supply (rural farming), Bass Pro (outdoor hunting), Hobby Lobby (craft/evangelical), Chick-fil-A (Southern evangelical), vs. Trader Joe's (urban educated), REI (outdoor progressive). These are corporate site selection decisions that already incorporate deep demographic/psychographic analysis.
- **Signal**: Consumer culture fingerprint as political identity. Corporate retail geography reflects decades of market research about who lives where — essentially pre-computed community type detection.
- **Cost/Effort**: Free via store locators + OSM. Requires scraping/API calls per brand. Medium effort for comprehensive coverage.

### 122. Chick-fil-A vs. Popeyes Density Ratio
- **What**: County-level ratio of Chick-fil-A to Popeyes locations — both are Southern chicken chains but with dramatically different customer demographics and cultural associations.
- **Resolution**: County | **Temporal**: Current
- **URL**: Store locators + OSM + Yelp Fusion API
- **Why it matters**: Chick-fil-A is one of the strongest cultural markers of white evangelical conservatism (closed Sundays, overt Christian corporate identity, 2012 Culture War appreciation day). Popeyes skews Black and urban. The ratio captures a cultural geography signal that traditional demographics describe as race+religion but that consumer choice reveals more naturally.
- **Signal**: Fast food chain geography as cultural community marker. The Chick-fil-A belt (suburban South) vs. Popeyes concentration (urban, Black communities) maps directly onto the political geography.
- **Cost/Effort**: Free (OSM + store locators). Low effort.

### 123. Waffle House Index / Density
- **What**: County-level Waffle House restaurant count and density — the "Waffle House Index" is FEMA's informal disaster response readiness proxy.
- **Resolution**: County | **Temporal**: Current + historical expansion
- **URL**: Waffle House store locator + FEMA references; Walter Hickey's analysis at FiveThirtyEight
- **Why it matters**: Waffle House is concentrated in the Southeast — 1,900+ locations, nearly all in the South. Its expansion geography traces the I-85/I-75/I-95 corridors through FL/GA/AL precisely. Waffle House density is a proxy for working-class, Southern, highway-corridor community culture. Its absence marks either extreme rural poverty (too sparse for any restaurant) or non-Southern metro areas.
- **Signal**: Working-class Southern cultural geography. Waffle House presence/absence marks the boundary of "the South" more precisely than any demographic variable.
- **Cost/Effort**: Free, store locator. ~1,900 locations, trivial to scrape.

### 124. Payday Lending & Title Loan Density (CFPB)
- **What**: County-level density of payday lenders, auto title loan shops, and check cashing stores. Includes CFPB complaint data on predatory lending by geography.
- **Resolution**: County | **Temporal**: CFPB data 2012-present; FDIC unbanked survey biennial
- **URL**: CFPB Complaint Database: https://www.consumerfinance.gov/data-research/consumer-complaints/ | FDIC National Survey of Unbanked/Underbanked Households | State licensing databases (FL OFR, GA DBF)
- **Why it matters**: Payday lending density maps directly onto financial vulnerability geography. FL has one of the highest payday loan volumes nationally. These businesses are legal but predatory, concentrated in low-income communities, and their density is a more sensitive poverty signal than SNAP because they capture the working-poor who earn too much for government assistance but too little for mainstream banking.
- **Signal**: Financial predation as community economic distress signal. Payday lender density distinguishes "working poor" communities from "welfare-dependent poor" communities — politically different populations with different grievances.
- **Cost/Effort**: Free (CFPB complaints, state licensing databases). Requires compilation.

### 125. AirBnB / VRBO Listing Density (AirDNA / Inside Airbnb)
- **What**: County-level short-term rental (STR) listing counts, average nightly rates, occupancy rates, and revenue estimates. Inside Airbnb publishes free quarterly snapshots for major US markets.
- **Resolution**: Point/ZIP (aggregatable to county) | **Temporal**: 2015-present
- **URL**: Inside Airbnb: http://insideairbnb.com/get-the-data/ (free, major cities) | AirDNA (paid for comprehensive county data, free public market reports)
- **Why it matters**: STR density is a tourism economy signal — counties with high AirBnB density have different economic structures (tourism-dependent, seasonal employment, housing cost pressure from investor purchases). FL's Gulf Coast, Panhandle beach towns, and Orlando metro have extremely high STR density. AirBnB-heavy counties also experience housing affordability crises that drive political attitudes.
- **Signal**: Tourism economy dependency and housing affordability stress. Coastal FL's STR economy creates communities where local workers can't afford to live in the area where they work — a political frustration signal.
- **Cost/Effort**: Free for Inside Airbnb (major markets only). Full county coverage requires AirDNA ($).

### 126. Chain Restaurant Diversity Index
- **What**: Count of unique chain restaurant brands per county, as a measure of commercial diversity/homogeneity. A county with only McDonald's, Subway, and Dollar General is structurally different from one with 40 unique chains.
- **Resolution**: County | **Temporal**: Current
- **URL**: OpenStreetMap POI data + Yelp Fusion API + Safegraph historical
- **Why it matters**: Commercial diversity is a proxy for economic complexity and population density. "Chain desert" counties (only 2-3 chains) are the most economically isolated. "Chain saturated" counties (suburban strip malls) represent a different community type. The presence of "emerging" chains (Raising Cane's, Wingstop, Boba tea shops) signals younger, growing communities.
- **Signal**: Economic complexity and commercial vitality through retail landscape. The chain composition tells you what corporate America thinks about the community.
- **Cost/Effort**: Moderate. OSM data is free but requires substantial POI processing.

---

### Financial & Economic Signals

### 127. Student Loan Debt Geography (NY Fed / CFPB)
- **What**: County-level aggregate student loan balances, delinquency rates, average per-borrower debt, and Public Service Loan Forgiveness (PSLF) application rates.
- **Resolution**: County (NY Fed Consumer Credit Panel) / ZIP (some CFPB data) | **Temporal**: 2003-present, quarterly
- **URL**: NY Fed Consumer Credit Panel (county aggregates): https://www.newyorkfed.org/microeconomics/hhdc | CFPB student lending data
- **Why it matters**: Student debt geography captures both education access and economic burden. Counties with high aggregate student debt are college-educated communities; counties with high delinquency rates are communities where education investment didn't pay off. The Biden student loan forgiveness debate had specific geographic patterns of political impact.
- **Signal**: Education investment burden. High-debt, high-delinquency counties (for-profit college victims, underemployed graduates) have specific political frustrations distinct from both the debt-free and the debt-positive.
- **Cost/Effort**: Free (NY Fed aggregates, CFPB). County-level, quarterly.

### 128. Cryptocurrency Adoption Proxy (ATM Locations / Exchange Signups)
- **What**: County-level Bitcoin ATM density + estimated cryptocurrency adoption from CoinATMRadar, CoinDesk surveys, and blockchain analytics firms.
- **Resolution**: Point/county | **Temporal**: 2017-present
- **URL**: CoinATMRadar: https://coinatmradar.com/api/ (free API, all US ATM locations); Bitcoin ATM locations also on OSM
- **Why it matters**: Bitcoin ATM density is concentrated in two distinct community types: tech-savvy libertarian areas AND underbanked immigrant communities (crypto as remittance tool). South FL has the highest Bitcoin ATM density in the US — in both Miami's tech corridor and Little Havana/Hialeah's Venezuelan community. This dual signal captures anti-institutional economic sentiment.
- **Signal**: Anti-institutional finance sentiment. Crypto geography captures libertarian economics AND immigrant remittance patterns — a rare variable that bridges two very different community types.
- **Cost/Effort**: Free, CoinATMRadar API. Point data, easy county aggregation.

### 129. GoFundMe / Crowdfunding Geography
- **What**: County-level density and category distribution of GoFundMe campaigns — medical fundraising (healthcare access proxy), emergency/disaster relief, memorial/funeral, education, community projects.
- **Resolution**: Estimated from GoFundMe search by location | **Temporal**: 2010-present
- **URL**: GoFundMe public campaign search (no bulk download; requires scraping by location)
- **Why it matters**: Medical GoFundMe campaigns are the clearest signal of healthcare system failure — communities where people crowdfund surgery/insulin/cancer treatment are communities where health insurance gaps are most severe. The ratio of medical:community:memorial campaigns reveals different community needs. Funeral GoFundMe density is a poverty signal.
- **Signal**: Community safety net adequacy as revealed by crowdfunding desperation. High medical GoFundMe density = healthcare access desert with a different political character than those captured by insurance rates alone.
- **Cost/Effort**: Moderate — no bulk data download. Requires geographic search scraping.

### 130. Lottery Ticket Sales by County (State Lottery Commissions)
- **What**: Per-capita lottery ticket sales by county — instant tickets (scratch-offs) vs. draw games (Powerball/Mega Millions), retailer density, and prize claim geography.
- **Resolution**: County | **Temporal**: Annual
- **URL**: FL Lottery: https://www.flalottery.com/retailerSearchResults | GA Lottery: https://www.galottery.com/ | AL has no state lottery (itself a political signal)
- **Why it matters**: Per-capita lottery spending is one of the strongest correlates of poverty and economic hopelessness — a "desperation tax." Counties with high scratch-off sales have populations that have given up on economic mobility through traditional means. AL's absence of a state lottery is itself a major political signal — repeated ballot failures due to evangelical opposition. The lottery referendum is one of AL's most politically revealing ballot measures.
- **Signal**: Economic hopelessness and risk-seeking behavior. Also: AL's anti-lottery politics as a direct measure of evangelical cultural governance power.
- **Cost/Effort**: Free (FL and GA lottery commissions publish retailer/county data). AL: N/A (no lottery).

---

### Family Structure & Demographics

### 131. Divorce Rate by County (CDC NVSS / State Vital Statistics)
- **What**: County-level marriage and divorce rates per 1,000 population. Includes median age at first marriage.
- **Resolution**: County | **Temporal**: Annual (CDC stopped county divorce data in 2000; state vital statistics continue)
- **URL**: CDC NVSS marriage/divorce data: https://www.cdc.gov/nchs/nvss/marriage-divorce.htm | FL Dept of Health vital statistics | GA DPH vital records
- **Why it matters**: Divorce rates capture family stability as a community characteristic. Paradoxically, the most conservative (evangelical, pro-family) states have among the highest divorce rates — the "red state divorce paradox." County-level variation reveals whether family stability rhetoric matches family stability reality. Early marriage + early divorce = a specific community pattern.
- **Signal**: Family structure instability as community characteristic. The gap between family values rhetoric and family stability reality is a political tension signal.
- **Cost/Effort**: Free (state vital statistics). Varies by state availability.

### 132. Single-Parent Household Rate by Race (ACS)
- **What**: County-level percentage of households with children that are single-parent, broken down by race and sex of householder.
- **Resolution**: Census tract | **Temporal**: Annual (ACS 1-year, 5-year)
- **URL**: ACS Table B11003 (family type), B09005 (household type for children under 18)
- **Why it matters**: Single-parent household rate (especially for children) is one of the strongest predictors of intergenerational poverty and has become a highly politicized variable. The racial disparity in single parenthood rates is a core conservative policy talking point. County-level variation within race groups captures structural factors beyond culture — employment, incarceration, military deployment.
- **Signal**: Family structure as both cause and consequence of economic opportunity. Single-parent rates interact with childcare desert data to predict women's economic autonomy.
- **Cost/Effort**: Free (ACS). Already available via our ACS pipeline; just need to add these tables.

### 133. Multigenerational Household Rate (ACS)
- **What**: Percentage of households containing 3+ generations (grandparent, parent, grandchild) by county, from ACS Table B11017.
- **Resolution**: County/tract | **Temporal**: Annual (ACS)
- **URL**: ACS Table B11017
- **Why it matters**: Multigenerational living captures cultural differences in family structure — higher in Hispanic, Asian, and immigrant communities, lower in white Anglo-Saxon communities. Also signals economic necessity (can't afford separate housing). This variable distinguishes immigrant-integrated communities from native-born communities in ways that ethnicity alone misses.
- **Signal**: Cultural family structure and economic compression. High multigenerational rate in non-immigrant areas signals housing affordability crisis.
- **Cost/Effort**: Free. Single ACS table. Trivial.

---

### Civic Infrastructure & Institutions

### 134. Civic Organization Membership (Rotary, Lions, Kiwanis, VFW, Elks)
- **What**: County-level membership counts for major civic organizations — Rotary International, Lions Club, Kiwanis, VFW, American Legion, Elks, Moose Lodge, Masonic lodges.
- **Resolution**: Club location (county mappable) | **Temporal**: Various
- **URL**: Rotary: https://www.rotary.org/en/search/clubs | Lions: https://www.lionsclubs.org/ | VFW: https://www.vfw.org/find-a-post | American Legion: https://www.legion.org/posts | Masonic lodges: state Grand Lodge directories
- **Why it matters**: Civic organization density is Putnam's "Bowling Alone" thesis operationalized. These organizations represent the declining civic infrastructure of small-town America. VFW/American Legion membership captures veteran community density AND civic engagement. Masonic lodge density historically correlated with community leadership infrastructure. The DECLINE in membership over time captures social capital erosion.
- **Signal**: Social capital and civic engagement infrastructure. Counties where these organizations still thrive have different political cultures from those where they've died. VFW/Legion membership is the strongest proxy for veteran community identity.
- **Cost/Effort**: Free (organization websites have club finders). Requires scraping multiple sources.

### 135. Volunteer Fire Department vs. Professional Fire Service
- **What**: Whether a county is served primarily by volunteer fire departments (VFDs), professional departments, or combination departments. VFD count and volunteer firefighter count per county.
- **Resolution**: County | **Temporal**: NFPA survey (periodic), USFA registry (annual)
- **URL**: USFA National Fire Department Registry: https://apps.usfa.fema.gov/registry/ | NFPA Fire Department Survey
- **Why it matters**: Volunteer fire departments are the backbone of rural community infrastructure — in many small towns, the VFD is the largest civic organization. VFD membership is a proxy for community volunteerism, social cohesion, and self-reliance culture. Professional fire service signals urban/suburban density and tax base. The transition from VFD to professional service often accompanies the loss of small-town identity.
- **Signal**: Rural civic infrastructure and self-reliance culture. VFD-dominant counties have a specific community identity tied to volunteer service and local mutual aid.
- **Cost/Effort**: Free, USFA registry. County-level, downloadable.

### 136. Religious Non-Profit Revenue (IRS 990 by Denomination Category)
- **What**: Total revenue of religious organizations (churches, ministries, parachurch organizations) by county, broken down by denomination category (evangelical, mainline, Catholic, non-Christian), from IRS 990 filings.
- **Resolution**: ZIP/county | **Temporal**: Annual
- **URL**: IRS 990 data (already listed as #21 for general nonprofits); ProPublica Nonprofit Explorer: https://projects.propublica.org/nonprofits/ | NCCS/Urban Institute national nonprofit data
- **Why it matters**: Goes beyond RCMS congregation counts to measure economic SCALE of religious institutions. A county with 50 small Baptist churches is different from one with 3 megachurches of equal total attendance. Megachurch revenue concentration signals a specific type of evangelical entrepreneurial culture. Parachurch ministry revenue (Focus on the Family-type orgs) signals culture war infrastructure investment.
- **Signal**: Religious economic power and institutional capacity. Megachurch-dominated communities have different political mobilization infrastructure from mainline-church communities.
- **Cost/Effort**: Free (IRS 990 bulk data). Requires NTEE code classification of religious orgs. Moderate effort.

---

### Education Signals

### 137. For-Profit College Enrollment Geography (IPEDS)
- **What**: County-level enrollment at for-profit institutions (University of Phoenix, DeVry, ITT Tech legacy, etc.) as share of total postsecondary enrollment.
- **Resolution**: Institution (county mappable) | **Temporal**: Annual (IPEDS)
- **URL**: NCES IPEDS: https://nces.ed.gov/ipeds/ (free data download)
- **Why it matters**: For-profit college enrollment is concentrated in low-income communities with limited access to public higher education. These students bear the worst student loan outcomes — high debt, low completion, poor job placement. For-profit college victims are a specific political constituency with grievances about both the education system and student debt policy.
- **Signal**: Predatory education exposure. Counties with high for-profit enrollment have populations that invested in education but got scammed — a specific political frustration distinct from either the college-educated or the non-college.
- **Cost/Effort**: Free (IPEDS). Institution-level, aggregatable to county. Annual.

### 138. Homeschool Rate by County (State Education Departments)
- **What**: Percentage of school-age children registered as homeschooled, by county.
- **Resolution**: County/district | **Temporal**: Annual
- **URL**: FL: https://www.fldoe.org/schools/school-choice/home-education/ | GA: Georgia Dept of Ed annual reports | AL: Board of Education (church school umbrella registration)
- **Why it matters**: Homeschool rates have surged post-pandemic and are concentrated in two distinct populations: religious conservatives (Christian curriculum, protection from secular values) and pandemic-era parents of all backgrounds. The pre-pandemic homeschool rate was ~3% nationally; post-pandemic it's ~6-11% depending on measurement. FL has the most permissive homeschool laws in the Southeast. High homeschool rates signal distrust of public institutions.
- **Signal**: Institutional distrust and cultural conservatism. Pre-pandemic homeschool rates are a strong religiosity signal; post-pandemic rates capture a broader anti-institutional shift.
- **Cost/Effort**: Free (state education departments). Availability varies by state.

### 139. School Voucher / Education Savings Account Participation (FL)
- **What**: County-level participation rates in FL's expanded school voucher / Education Savings Account (ESA) programs — the most expansive in the nation post-2023.
- **Resolution**: County | **Temporal**: 2023-present (universal expansion)
- **URL**: FL Dept of Education Step Up for Students program data; FL DOE choice scholarship reports
- **Why it matters**: FL's universal ESA program allows ANY family to use public funds for private school. Participation rates reveal where families are opting out of public education — a direct measure of public school dissatisfaction AND financial capacity to exercise choice. High voucher uptake in evangelical communities signals the school choice + religious education nexus.
- **Signal**: Public education exit rate as political identity. Counties with high voucher uptake are politically invested in the school choice agenda — a core Republican policy priority in FL.
- **Cost/Effort**: Free (FL DOE). County-level, annual.

---

### Health & Substance Signals

### 140. Vaccine Exemption Rates — Pre-COVID (School Entry Immunizations)
- **What**: County-level rates of non-medical (philosophical/religious) vaccine exemptions for school-entry immunizations (MMR, DTaP, etc.), from state immunization programs.
- **Resolution**: School district/county | **Temporal**: Annual, 2005-present
- **URL**: CDC School Vaccination Assessment: https://www.cdc.gov/schoolvaccination/ | FL DOH county immunization reports | GA DPH school surveys
- **Why it matters**: Pre-COVID vaccine exemption rates (2015-2019) capture anti-vaccine sentiment BEFORE COVID politicized vaccination. These rates predict COVID vaccine refusal with high accuracy. Counties with high pre-COVID exemptions had established anti-institutional health attitudes independent of partisan identity. The correlation between pre-COVID exemptions and 2020 Trump vote is weaker than the COVID vaccine correlation — suggesting COVID vaccination captured a new political signal beyond pre-existing anti-vax sentiment.
- **Signal**: Pre-existing anti-institutional health attitudes vs. post-COVID politically-driven vaccine refusal. The divergence between the two signals is itself informative about community political dynamics.
- **Cost/Effort**: Free (CDC + state DOH). County or school-district level.

### 141. Methamphetamine Lab Seizures (DEA EPIC)
- **What**: County-level meth lab seizure counts — a direct measure of local meth production (distinct from opioid imports).
- **Resolution**: County | **Temporal**: 2004-2020 (DEA National Clandestine Laboratory Register)
- **URL**: DEA EPIC (El Paso Intelligence Center) meth lab database; some data published via ONDCP
- **Why it matters**: Meth is distinct from opioids — meth is locally produced in rural areas while opioids are prescribed or imported. Rural meth lab geography maps onto a specific community type: severely economically depressed white rural communities. AL and North FL have historically high meth lab seizure rates. The meth crisis predates the opioid crisis and affects a different (though overlapping) community type.
- **Signal**: Deep rural economic despair expressed through local drug production. Meth lab counties are the most extreme end of rural decline — more severe than opioid-affected communities, which can include suburban areas.
- **Cost/Effort**: Free (DEA publishes county data through ONDCP). Some years require FOIA.

### 142. Cannabis Dispensary Density (State Programs)
- **What**: Medical marijuana dispensary locations and sales volume by county, in states with legal medical/recreational cannabis.
- **Resolution**: Point/county | **Temporal**: State-specific (FL medical since 2016)
- **URL**: FL OMMU: https://knowthefactsmmj.com/registry/ | GA: limited program (no dispensaries as of 2024); AL: medical program launching
- **Why it matters**: FL's medical marijuana program was approved by 71% of voters in 2016 — massively bipartisan support. But dispensary placement geography reveals where the industry has invested. Dispensary density in FL correlates with population density and income, not the 71% approval rate. Counties that voted for medical marijuana but have zero dispensaries are experiencing an access disparity. The recreational legalization ballot question (failed 2024 at 56%, short of 60% threshold) had geographic patterns that reveal community attitudes toward cannabis.
- **Signal**: Drug policy attitude as community cultural signal. Cannabis support transcends partisanship in FL — but dispensary access geography reveals economic stratification of the legal market.
- **Cost/Effort**: Free (FL OMMU registry). Point data, county aggregation. AL/GA limited.

---

### Transportation & Geography

### 143. Interstate Highway Access & Distance to Major Metro
- **What**: County-level metrics: distance to nearest interstate highway interchange, drive time to nearest metro of 50K+ population, number of highway lane-miles, and truck traffic volume (FHWA HPMS).
- **Resolution**: County | **Temporal**: Annual (HPMS)
- **URL**: FHWA Highway Performance Monitoring System: https://www.fhwa.dot.gov/policyinformation/hpms.cfm | FHWA Travel Monitoring Analysis System
- **Why it matters**: Highway access is the primary determinant of economic connectivity for rural counties. Counties on interstates have different economic trajectories than counties bypassed by the highway system. Drive time to the nearest metro captures isolation — a 15-minute vs. 90-minute drive to a city fundamentally shapes job access, healthcare access, and cultural exposure.
- **Signal**: Geographic isolation as community defining feature. The most politically extreme rural communities are often the most geographically isolated. Highway bypass towns experience specific economic decline.
- **Cost/Effort**: Free (FHWA). County-level, some GIS computation needed.

### 144. Airport Passenger Volume & Connectivity
- **What**: County-level domestic and international passenger enplanements, direct flight destinations, and seasonal variation from FAA T-100 data.
- **Resolution**: Airport (county mappable) | **Temporal**: Monthly (FAA T-100)
- **URL**: FAA Air Carrier Activity Information System: https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/
- **Why it matters**: Airport connectivity captures economic class and cosmopolitan exposure. Counties with international airport access have populations exposed to global travel; counties with only regional airports or no airport have fundamentally different global connectivity. FL's tourism economy creates unusually high airport traffic in destination counties (Orlando, Miami, Fort Myers) — where airport workers form a distinct political constituency.
- **Signal**: Global connectivity and tourism economy. Airport-adjacent communities have specific economic dependence (TSA, airline, rental car workers) with distinct labor politics.
- **Cost/Effort**: Free (FAA). Airport-level, county aggregation easy.

---

### Social Media & Digital Behavior

### 145. Twitter/X Political Bot Activity by Geography (Bot Sentinel / Academic Datasets)
- **What**: Geographic distribution of suspected political bot/troll accounts and their engagement patterns, from academic bot detection research and Bot Sentinel archives.
- **Resolution**: DMA/state | **Temporal**: 2016-2023
- **URL**: Bot Sentinel: https://botsentinel.com/ | Academic: Varol et al. "Online Human-Bot Interactions" (Indiana University); Twitter Research API academic archives
- **Why it matters**: Bot-amplified political content has geographic targeting — foreign influence operations and domestic astroturfing concentrate on swing states and counties. FL and GA were top targets for Russian IRA operations in 2016. The degree to which a community's online discourse is bot-influenced affects information quality and polarization.
- **Signal**: Information environment corruption. Counties most exposed to bot-amplified content may show disproportionate polarization or conspiracy belief adoption.
- **Cost/Effort**: Academic access required. Some Bot Sentinel data publicly available.

### 146. Nextdoor Activity & Community Engagement (Public Data)
- **What**: Nextdoor neighborhood group activity levels, local government engagement, and community concern categories (crime, politics, recommendations) by neighborhood/ZIP.
- **Resolution**: Neighborhood/ZIP | **Temporal**: 2015-present
- **URL**: Nextdoor API (limited public data); academic research using Nextdoor data
- **Why it matters**: Nextdoor is the most locally-focused social network — users are verified by address. Political activity on Nextdoor captures hyperlocal political attitudes (yard sign disputes, school board debates, policing concerns) that other social media miss. Nextdoor's "Crime & Safety" category activity correlates with fear-based political attitudes.
- **Signal**: Hyperlocal political anxiety and community concern priorities. High "Crime & Safety" posting activity may predict tough-on-crime political preferences.
- **Cost/Effort**: Limited — Nextdoor restricts API access. Academic research papers provide some geographic analysis.

---

### Government Service & Dependency

### 147. Social Security Retirement Dependency Ratio (SSA)
- **What**: County-level ratio of Social Security retirement beneficiaries to working-age population, plus average monthly benefit amount and total county Social Security income.
- **Resolution**: County | **Temporal**: Annual
- **URL**: SSA OASDI Beneficiary Statistics: https://www.ssa.gov/policy/docs/statcomps/oasdi_sc/ | SSA County-Level Data (annual)
- **Why it matters**: Retirement community concentration defines entire county economies in FL. The Villages (Sumter County), Cape Coral (Lee County), and much of FL's Gulf Coast are economically dependent on Social Security income. These communities have specific political interests: protecting Social Security, Medicare, and property tax exemptions. The retiree share of voters is disproportionate due to higher turnout.
- **Signal**: Retiree political power concentration. High SS dependency ratio = communities where retirees dominate the electorate AND the economy. Their political priorities (healthcare, SS, taxes on fixed income) differ from working-age communities.
- **Cost/Effort**: Free (SSA annual downloads). County-level, trivial.

### 148. Federal Government Employee Concentration (OPM FedScope)
- **What**: County-level count of federal civilian employees by agency, pay grade, and occupation — from OPM's FedScope data cube.
- **Resolution**: County (duty station) | **Temporal**: Quarterly, 2000-present
- **URL**: OPM FedScope: https://www.fedscope.opm.gov/ (free, interactive + bulk download)
- **Why it matters**: Federal employee concentration defines specific community types: military base counties (DoD civilians), VA hospital counties, federal courthouse communities, and DC-adjacent commuter counties. These communities have direct economic stake in government spending and distinct political attitudes toward government efficiency vs. downsizing (DOGE, sequestration).
- **Signal**: Direct government employment dependency. Federal employee-heavy counties shift differently during government shutdown/sequestration periods. FL has major DoD civilian concentrations (MacDill, Eglin, Patrick SFB).
- **Cost/Effort**: Free (OPM FedScope). County-level, quarterly. Very clean data.

### 149. State & Local Government Pension Obligations (Pew / Census)
- **What**: County or state-level pension funding ratios (assets vs. liabilities), pension benefit generosity, and retired government employee count.
- **Resolution**: State/pension system (some county data) | **Temporal**: Annual
- **URL**: Pew Charitable Trusts pension tracker; Census of Governments (pension employment data); state pension system annual reports (FL FRS, GA ERS, AL RSA)
- **Why it matters**: Pension obligation crisis drives local fiscal politics — counties with underfunded pensions face tax increases or service cuts. FL's FRS is one of the best-funded state pensions (78%); AL's RSA is significantly underfunded. Local government pension stress directly affects property taxes, school funding, and public services — all politically salient.
- **Signal**: Fiscal sustainability and intergenerational government obligation. Pension-stressed counties have specific tax-revolt political dynamics.
- **Cost/Effort**: Free (Pew tracker + Census). State-level primary; some county-level pension data available.

---

### Migration & Population Dynamics

### 150. Snowbird Migration Patterns (USPS + FL DMV)
- **What**: Seasonal population fluctuation in FL counties from "snowbird" migration — estimated from USPS seasonal address changes, FL DMV seasonal registrations, and utility connection patterns.
- **Resolution**: County/ZIP | **Temporal**: Annual seasonal cycle (November-April peak)
- **URL**: HUD/USPS vacancy data (#22 already listed); FL HSMV vehicle registration seasonal spikes; FL Power & Light seasonal connection data (rate case filings)
- **Why it matters**: FL's political composition literally changes by season. Snowbird population (primarily from Northeastern and Midwestern states) brings different political attitudes to FL's Gulf Coast, South FL, and retirement communities. Some snowbirds become FL residents (shifting voter registration), while seasonal visitors affect community culture without voting. The conversion rate from snowbird to permanent resident is a demographic shift signal.
- **Signal**: Seasonal political culture mixing and permanent migration pipeline. Counties with high snowbird flux are communities in transition — the snowbird-to-resident conversion rate predicts future political shift direction.
- **Cost/Effort**: Moderate — requires combining USPS, DMV, and utility data. No single source captures this.

### 151. Puerto Rican Migration Post-Hurricane Maria (2017)
- **What**: County-level estimates of Puerto Rican arrivals after Hurricane Maria (Sept 2017) from FEMA registration data, school enrollment surges, and ACS migration supplements.
- **Resolution**: County | **Temporal**: 2017-2020
- **URL**: Center for Puerto Rican Studies (Hunter College): https://centropr.hunter.cuny.edu/ — detailed post-Maria migration estimates | FEMA Individual Assistance data (county-level registrations)
- **Why it matters**: ~130,000 Puerto Ricans relocated to FL after Hurricane Maria, concentrated in Orange, Osceola, and Hillsborough counties. This migration reshaped FL's I-4 corridor politics — Puerto Ricans are US citizens who can immediately register to vote. The 2018 FL governor race (DeSantis won by 0.4%) may have been affected by this migration. Post-Maria arrivals have distinct political attitudes from long-established FL Puerto Rican communities.
- **Signal**: Disaster-driven migration as political demographic shock. The Maria diaspora is a natural experiment in rapid community composition change with direct electoral consequences.
- **Cost/Effort**: Free (Centro PR estimates, FEMA data). County-level.

### 152. Venezuelan Exile Community Concentration
- **What**: County-level concentration of Venezuelan-born residents, from ACS country-of-birth tables. Includes naturalization rates and arrival decade.
- **Resolution**: County (ACS B05006) | **Temporal**: Annual (ACS 1-year for large counties)
- **URL**: ACS Table B05006 (place of birth, foreign-born) | USCIS naturalization statistics by country of origin
- **Why it matters**: Venezuelan exiles in South FL (Doral, Weston, Homestead) are among the most strongly Republican Hispanic communities — driven by anti-socialist/anti-Maduro sentiment. Their political behavior defies the "Hispanic = Democratic" assumption. Doral, FL is informally called "Doralzuela." The Venezuelan community's political intensity (anti-communism, libertarian economics) shapes the entire South FL political landscape.
- **Signal**: Anti-communist exile politics as community type. Venezuelan concentration distinguishes South FL's Hispanic politics from Central FL's Puerto Rican and immigrant Mexican communities. This is a key intra-Hispanic political fault line.
- **Cost/Effort**: Free (ACS). Already in our ACS pipeline; need country-of-birth tables.

---

### Land Use & Built Environment

### 153. Mobile Home / Manufactured Housing Density (ACS + HUD)
- **What**: Percentage of housing units that are mobile homes / manufactured housing by county, from ACS housing characteristics tables.
- **Resolution**: County/tract | **Temporal**: Annual (ACS)
- **URL**: ACS Table B25024 (units in structure); HUD manufactured housing surveys
- **Why it matters**: Mobile home concentration is among the strongest single-variable predictors of white rural poverty and Republican lean. FL has the highest mobile home count of any state (~850K units). Mobile home parks in rural FL/AL/GA represent a specific community type: white working-class, low-wealth, vulnerable to extreme weather, and often on leased land (no land ownership). Mobile home residents are a distinct political constituency from homeowners or apartment renters.
- **Signal**: Housing vulnerability and white rural poverty. Mobile home density predicts Trump vote share at the county level better than median income in FL/AL.
- **Cost/Effort**: Free (ACS). Already available in our pipeline; single table.

### 154. Gated Community & HOA Density
- **What**: Estimated number of homes within gated communities and homeowners associations (HOAs) by county, from Census AHS and Community Associations Institute (CAI) data.
- **Resolution**: County (estimated) | **Temporal**: American Housing Survey (biennial), CAI annual reports
- **URL**: Census American Housing Survey: https://www.census.gov/programs-surveys/ahs.html | Community Associations Institute: https://www.caionline.org/
- **Why it matters**: FL has more HOA-governed homes than any other state (~9 million residents in HOAs). Gated community residence signals economic class, desire for social control, and NIMBYism. The Villages is the world's largest gated retirement community and one of the most Republican-voting communities in FL. HOA governance creates a specific political culture: rule-following, property-value-protective, aesthetically conformist.
- **Signal**: Private governance as community identity. HOA-dominant communities have specific political preferences around property rights, zoning, and local control that differ from non-HOA areas.
- **Cost/Effort**: Moderate. CAI publishes state-level estimates; county-level requires AHS or property record analysis.

### 155. New Housing Construction Type (Census Building Permits)
- **What**: County-level building permits by structure type: single-family detached, single-family attached (townhomes), 2-4 unit, 5+ unit multifamily. Monthly data.
- **Resolution**: County | **Temporal**: Monthly, 2004-present
- **URL**: Census Building Permits Survey: https://www.census.gov/construction/bps/ (already referenced in #7 — this entry focuses on TYPE of construction)
- **Why it matters**: What TYPE of housing is being built reveals the community's future trajectory. Single-family detached construction signals suburban/exurban sprawl and family-formation communities. 5+ multifamily signals urban densification and renter communities. FL's construction boom is heavily tilted toward single-family in exurbs (St. Johns, Pasco, Manatee counties) — these are the communities that are politically shifting fastest.
- **Signal**: Community trajectory through construction decisions. The ratio of multifamily to single-family construction predicts whether a community is urbanizing or suburbanizing — and the political implications of each.
- **Cost/Effort**: Free (Census). County-level, monthly. Already partially covered by #7 but this focuses on TYPE.

---

### Political Infrastructure & Campaigns

### 156. Super PAC Spending by Media Market (FEC IE Database)
- **What**: Independent expenditure (IE) spending by Super PACs and 501(c)(4)s, allocated to media markets (DMA) where the ads air, for/against specific candidates.
- **Resolution**: DMA/state | **Temporal**: 2010-present (Citizens United era)
- **URL**: FEC Independent Expenditures: https://www.fec.gov/data/independent-expenditures/ | Wesleyan Media Project: https://mediaproject.wesleyan.edu/ (academic ad tracking)
- **Why it matters**: Where Super PACs spend money reveals where they think elections can be influenced. Heavy IE spending in a county/DMA signals the political establishment views it as a battleground. The TYPES of ads (attack vs. positive, issues emphasized) reveal what political operatives think the community responds to.
- **Signal**: Campaign resource allocation as elite judgment of community political elasticity. Super PAC spending geography is essentially professional community political analysis — where they spend is where they think voters are persuadable.
- **Cost/Effort**: Free (FEC). DMA allocation requires matching spending to ad buy markets.

### 157. Campaign Rally & Event Locations (Campaign Trail Tracker)
- **What**: Geocoded locations of presidential and senatorial campaign rally/event appearances, with crowd size estimates and event type (rally, town hall, fundraiser, church visit).
- **Resolution**: Point (county mappable) | **Temporal**: 2008-2024
- **URL**: FairVote campaign trail tracking; academic datasets (e.g., Shaw 2006 "The Race to 270"); media compilations
- **Why it matters**: Where candidates hold rallies reveals their strategic assessment of community political value. Trump's rally geography (large venues in exurban areas) vs. Harris/Biden rally geography (colleges, Black churches, union halls) maps onto community type targeting. FL's I-4 corridor rally density is the highest in the nation for presidential campaigns.
- **Signal**: Campaign strategic targeting reveals professional assessment of community political characteristics. Rally-visited communities are perceived as persuadable or mobilizable.
- **Cost/Effort**: Free (media tracking, academic datasets). Some manual compilation needed.

### 158. Local Party Committee Activity (FEC State/Local Party Reports)
- **What**: Quarterly financial reports of county-level and state-level Democratic and Republican party committees — fundraising, expenditures, volunteer activity, and staff count.
- **Resolution**: County/district | **Temporal**: Quarterly, 2000-present
- **URL**: FEC Committee Reports: https://www.fec.gov/data/committee/ (search by committee type = "Party - State/Local") | State campaign finance databases (FL DSFE, GA Ethics Commission)
- **Why it matters**: Local party committee financial health is a direct measure of partisan organizational infrastructure. Counties where the Democratic Party has no funded committee vs. where it has a 6-figure budget behave differently — regardless of vote share. Party committee activity captures volunteer networks, voter contact capacity, and candidate recruitment infrastructure.
- **Signal**: Partisan organizational capacity. Dormant local parties signal communities that are politically uncontested — and may show different shift dynamics than communities with active two-party competition.
- **Cost/Effort**: Free (FEC + state databases). Requires querying by committee type and geography.

---

### Miscellaneous High-Signal Sources

### 159. Pet Ownership Demographics (AVMA / Simmons National Consumer Survey)
- **What**: County/metro-level estimates of dog vs. cat ownership rates, pet spending, and veterinary clinic density.
- **Resolution**: Metro/state | **Temporal**: AVMA survey (every 5 years)
- **URL**: AVMA Pet Ownership & Demographics Sourcebook; Simmons National Consumer Survey (academic access); OSM veterinary clinic POI data
- **Why it matters**: This sounds frivolous but is remarkably robust: the dog:cat ownership ratio at the county level correlates with political lean (R² ~0.3 with Republican vote share in some analyses). Dog ownership correlates with homeownership, suburban/rural living, family formation, and conscientiousness. Cat ownership correlates with urban, single, renter demographics. This is a revealed-preference lifestyle signal that captures community character without political questions.
- **Signal**: Lifestyle proxy for community character. The dog:cat ratio is a legitimate consumer research variable used in market segmentation — essentially pre-computed community typing from an unexpected angle.
- **Cost/Effort**: Limited free data (AVMA report is $$$; veterinary clinic density from OSM is free proxy). Metro-level estimates from Simmons accessible via academic libraries.

### 160. Blood Donation Rates (Red Cross / Blood Center Regional Data)
- **What**: County or region-level blood donation rates and blood drive frequency from American Red Cross and independent blood centers.
- **Resolution**: Blood center service area (~county) | **Temporal**: Annual
- **URL**: American Red Cross regional reports; independent blood centers (OneBlood in FL, Lifesouth in GA/AL) publish collection data; AABB national blood collection surveys
- **Why it matters**: Blood donation is one of the cleanest measures of prosocial civic behavior — it requires time, physical effort, and no personal benefit. Donation rates capture community altruism and civic engagement. Disaster-response blood drive surges capture community solidarity. OneBlood (FL) publishes county-level collection statistics.
- **Signal**: Community prosocial behavior and civic health. Blood donation rates may correlate with social capital measures but capture a different dimension — willingness to contribute bodily resources for strangers.
- **Cost/Effort**: Moderate — blood center regional data requires compilation. OneBlood FL data most accessible.

### 161. Organ Donor Registration Rates (State DMV Data)
- **What**: Percentage of licensed drivers who have registered as organ donors by county, from state DMV/driver license records.
- **Resolution**: County | **Temporal**: Annual
- **URL**: FL HSMV organ donor registration data; Donate Life America annual report; UNOS (organ allocation)
- **Why it matters**: Organ donor registration rates capture institutional trust, bodily autonomy attitudes, and altruistic orientation. Counties with low registration rates may reflect distrust of medical institutions (especially in communities with historical medical abuse experiences) or religious beliefs about bodily integrity.
- **Signal**: Institutional trust and medical system confidence. Low organ donation registration may predict vaccine hesitancy and other anti-institutional health behaviors.
- **Cost/Effort**: Free (Donate Life America aggregates; FL HSMV may publish county data). Low effort.

### 162. Daycare Center vs. Family Daycare Ratio
- **What**: County-level ratio of licensed commercial daycare centers to family/home-based daycare providers, from state childcare licensing databases.
- **Resolution**: County | **Temporal**: Annual
- **URL**: FL DCF child care licensing database; GA DECAL; AL DHR child care search
- **Why it matters**: The ratio of center-based to family-based childcare reflects both economic opportunity (center-based requires more capital) and cultural preference (family-based daycare is preferred in tight-knit communities where neighbors care for each other's children). Rural communities with only family daycare have different community structures than suburban areas with commercial childcare chains.
- **Signal**: Community childcare infrastructure and cultural care norms. Family daycare dominance signals close-knit community networks; center dominance signals commercial/suburban development.
- **Cost/Effort**: Free (state licensing databases). Point data, county aggregation.

### 163. Cemetery Density & Type (VA National Cemetery vs. Private)
- **What**: County-level count and type of cemeteries: VA national cemeteries, church cemeteries, municipal cemeteries, private commercial cemeteries, and historical (pre-1900) cemeteries.
- **Resolution**: Point/county | **Temporal**: Current + historical
- **URL**: VA National Cemetery Administration: https://www.cem.va.gov/cems/ | USGS Geographic Names Information System (GNIS) cemetery feature class | FindAGrave.com
- **Why it matters**: Cemetery density captures community age and deep-rootedness. Counties with many small church cemeteries are long-established, multi-generational communities. VA cemetery presence signals military community. Commercial cemetery growth signals new population without established community ties. Historical cemetery preservation (or neglect) signals community relationship to its own past.
- **Signal**: Community historical depth and rootedness. Deep-rooted communities (many old cemeteries) have different political identities than transient/new communities.
- **Cost/Effort**: Free (GNIS, VA data). GNIS has nationwide cemetery POI data.

---

### Updated Priority Matrix — Round 4 Additions

| Source | # | Category | Expected Signal | Effort |
|--------|---|----------|----------------|--------|
| Confederate Monuments | 119 | Extremism/Culture | Very High | Very Low |
| Cracker Barrel vs Whole Foods | 121 | Consumer | Very High | Medium |
| Payday Lending Density | 124 | Financial | High | Low |
| Student Loan Debt | 127 | Financial | High | Low |
| Lottery Sales | 130 | Economic | High | Low |
| Homeschool Rate | 138 | Education | High | Medium |
| Pre-COVID Vaccine Exemptions | 140 | Health | High | Low |
| Mobile Home Density | 153 | Land Use | Very High | Very Low (ACS) |
| SS Retirement Dependency | 147 | Government | High | Very Low |
| Federal Employee Concentration | 148 | Government | High | Very Low |
| Puerto Rican Post-Maria | 151 | Migration | Very High (FL) | Low |
| Venezuelan Exile Concentration | 152 | Migration | Very High (FL) | Very Low (ACS) |
| Fox/MSNBC Ratings by DMA | 114 | Media | Very High | Moderate |
| SPLC Hate Groups | 118 | Extremism | High | Very Low |
| AirBnB Density | 125 | Consumer | High | Low-Medium |
| Cannabis Dispensary | 142 | Cultural | Medium-High | Low |
| Campaign Rally Locations | 157 | Political | Medium | Low |
| Local Party Committee Activity | 158 | Political | High | Low |

---

*End of Expansion Round 4. Total ideated sources: 163 (113 prior + 50 new).*
