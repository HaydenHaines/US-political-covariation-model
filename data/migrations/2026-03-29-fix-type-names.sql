-- Type name corrections applied 2026-03-29
-- Fixes: wrong race labels (Majority-Black), wrong urbanity (Urban/Suburban), 
--        wrong Asian label, ordinal suffix disambiguation artifacts (II/III/etc.)
-- Applied directly to data/wethervane.duckdb types.display_name column

UPDATE types SET display_name = 'Black-Belt Evangelical Mixed'       WHERE type_id = 34;  -- was 'Majority-Black Urban' (53% White, 37% Black, rural)
UPDATE types SET display_name = 'Black-Belt Deep-Rural Evangelical'  WHERE type_id = 37;  -- was 'Majority-Black Evangelical Churched' (51% White, 39% Black)
UPDATE types SET display_name = 'Black-Belt Evangelical Devout'      WHERE type_id = 79;  -- was 'Majority-Black Evangelical Devout' (48% White, 42% Black)
UPDATE types SET display_name = 'Black-Belt Exurban Evangelical'     WHERE type_id = 23;  -- was 'Black-Belt Urban' (56/sqmi, not urban)
UPDATE types SET display_name = 'Sunbelt Catholic Retiree'           WHERE type_id = 30;  -- was 'Urban Retiree' (borderline log_dens=1.9)
UPDATE types SET display_name = 'Black-Belt Evangelical Low-Income'  WHERE type_id = 66;  -- was 'Black-Belt Suburban Black-Church' (26/sqmi, not suburban)
UPDATE types SET display_name = 'Rural Educated Young'               WHERE type_id = 88;  -- was 'Asian Rural Younger' (54% White, only 9% Asian)
UPDATE types SET display_name = 'Rural Mainline-Heavy Retiree'       WHERE type_id = 39;  -- was 'Rural Mainline Retiree Homeowner III'
UPDATE types SET display_name = 'Rural Mainline Retiree Static'      WHERE type_id = 65;  -- was 'Rural Mainline Retiree Homeowner II'
UPDATE types SET display_name = 'Rural Unchurched Evangelical-Lean'  WHERE type_id = 32;  -- was 'Rural Mainline Unchurched Older III'
UPDATE types SET display_name = 'Rural Unchurched Higher-Income'     WHERE type_id = 70;  -- was 'Rural Mainline Unchurched Older II'
UPDATE types SET display_name = 'Alaska Frontier Homeowner'          WHERE type_id = 17;  -- was 'Rural Unchurched Young Homeowner' (AK census areas)
UPDATE types SET display_name = 'Alaska Frontier Deep-Rural A'       WHERE type_id = 38;  -- was 'Rural Unchurched Young Homeowner II'
UPDATE types SET display_name = 'Alaska Frontier Deep-Rural B'       WHERE type_id = 41;  -- was 'Rural Unchurched Young Homeowner III'
UPDATE types SET display_name = 'Alaska Census Area I'               WHERE type_id = 51;  -- was 'Rural Unchurched Young Homeowner VI'
UPDATE types SET display_name = 'Alaska Census Area II'              WHERE type_id = 78;  -- was 'Rural Unchurched Young Homeowner IV'
UPDATE types SET display_name = 'Alaska Frontier Mixed'              WHERE type_id = 85;  -- was 'Rural Unchurched Young Homeowner V'
UPDATE types SET display_name = 'Alaska Census Area III'             WHERE type_id = 93;  -- was 'Rural Unchurched Young Homeowner VII'
UPDATE types SET display_name = 'Alaska Frontier Renter'             WHERE type_id = 19;  -- was 'Rural Unchurched Young Renter'
UPDATE types SET display_name = 'Alaska Census Area IV'              WHERE type_id = 46;  -- was 'Rural Unchurched Young Renter III'
UPDATE types SET display_name = 'Alaska Census Area V'               WHERE type_id = 54;  -- was 'Rural Unchurched Young Renter IV'
UPDATE types SET display_name = 'Alaska Census Area VI'              WHERE type_id = 68;  -- was 'Rural Unchurched Young Renter V'
UPDATE types SET display_name = 'Alaska Census Area VII'             WHERE type_id = 81;  -- was 'Rural Unchurched Young Renter II'
UPDATE types SET display_name = 'Alaska Census Area VIII'            WHERE type_id = 90;  -- was 'Rural Unchurched Young Renter VI'
