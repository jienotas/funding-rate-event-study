$ErrorActionPreference = 'Stop'

$outDir = Join-Path $PSScriptRoot '..\output'
$dataDir = Join-Path $PSScriptRoot '..\data'
New-Item -ItemType Directory -Force -Path $outDir, $dataDir | Out-Null

$cases = @(
    @{ venue = 'Binance'; asset = 'BTC'; instrument = 'BTCUSDT'; url = 'https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=100'; shape = 'binance' },
    @{ venue = 'Binance'; asset = 'ETH'; instrument = 'ETHUSDT'; url = 'https://fapi.binance.com/fapi/v1/fundingRate?symbol=ETHUSDT&limit=100'; shape = 'binance' },
    @{ venue = 'OKX'; asset = 'BTC'; instrument = 'BTC-USDT-SWAP'; url = 'https://www.okx.com/api/v5/public/funding-rate-history?instId=BTC-USDT-SWAP&limit=100'; shape = 'okx' },
    @{ venue = 'OKX'; asset = 'ETH'; instrument = 'ETH-USDT-SWAP'; url = 'https://www.okx.com/api/v5/public/funding-rate-history?instId=ETH-USDT-SWAP&limit=100'; shape = 'okx' },
    @{ venue = 'Bybit'; asset = 'BTC'; instrument = 'BTCUSDT'; url = 'https://api.bybit.com/v5/market/funding/history?category=linear&symbol=BTCUSDT&limit=100'; shape = 'bybit' },
    @{ venue = 'Bybit'; asset = 'ETH'; instrument = 'ETHUSDT'; url = 'https://api.bybit.com/v5/market/funding/history?category=linear&symbol=ETHUSDT&limit=100'; shape = 'bybit' }
)

$retrievedAt = [DateTimeOffset]::UtcNow.ToString('o')
$records = [System.Collections.Generic.List[object]]::new()
$raw = [System.Collections.Generic.List[object]]::new()

foreach ($case in $cases) {
    $response = Invoke-RestMethod -Uri $case.url -Method Get -TimeoutSec 30
    switch ($case.shape) {
        'binance' { $rows = @($response) }
        'okx' { $rows = @($response.data) }
        'bybit' { $rows = @($response.result.list) }
    }
    $raw.Add([PSCustomObject]@{ venue = $case.venue; asset = $case.asset; instrument = $case.instrument; source_url = $case.url; row_count = $rows.Count; response = $response })

    foreach ($row in $rows) {
        if ($case.shape -eq 'bybit') {
            $stamp = [int64]$row.fundingRateTimestamp
            $rate = $row.fundingRate
            $rateField = 'fundingRate (historical record)'
        } else {
            $stamp = [int64]$row.fundingTime
            $rate = $row.fundingRate
            $rateField = 'fundingRate (historical record)'
        }
        $utc = [DateTimeOffset]::FromUnixTimeMilliseconds($stamp).ToUniversalTime()
        $records.Add([PSCustomObject]@{
            venue = $case.venue
            asset = $case.asset
            instrument = $case.instrument
            funding_time_unix_ms = $stamp
            funding_time_utc = $utc.ToString('yyyy-MM-ddTHH:mm:ss.fffZ')
            funding_minute_unix_ms = [int64]([math]::Floor($stamp / 60000) * 60000)
            funding_time_utc_minute = $utc.ToString('yyyy-MM-ddTHH:mmZ')
            utc_hour = $utc.Hour
            utc_minute = $utc.Minute
            funding_rate = $rate
            rate_field_semantics = $rateField
            retrieved_at_utc = $retrievedAt
            source_url = $case.url
        })
    }
}

$records | Sort-Object asset, funding_time_unix_ms, venue | Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $dataDir '第十三周官方历史结算记录.csv')
$raw | ConvertTo-Json -Depth 12 | Set-Content -Encoding utf8 (Join-Path $dataDir '第十三周官方API原始响应.json')

$venueAsset = foreach ($group in ($records | Group-Object venue, asset)) {
    $items = @($group.Group)
    $times = @($items | ForEach-Object { [int64]$_.funding_time_unix_ms } | Sort-Object -Unique)
    $gaps = for ($i = 1; $i -lt $times.Count; $i++) { [math]::Round(($times[$i] - $times[$i - 1]) / 3600000, 3) }
    $hours = @($items | Group-Object utc_hour | Sort-Object { [int]$_.Name } | ForEach-Object { '{0}: {1}' -f $_.Name, $_.Count }) -join '; '
    [PSCustomObject]@{
        venue_asset = $group.Name
        row_count = $items.Count
        distinct_event_times = $times.Count
        first_utc = [DateTimeOffset]::FromUnixTimeMilliseconds($times[0]).ToString('yyyy-MM-ddTHH:mm:ssZ')
        last_utc = [DateTimeOffset]::FromUnixTimeMilliseconds($times[-1]).ToString('yyyy-MM-ddTHH:mm:ssZ')
        utc_hour_counts = $hours
        distinct_gaps_hours = (@($gaps | Sort-Object -Unique) -join ', ')
    }
}
$venueAsset | Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $outDir '第十三周交易所资产覆盖摘要.csv')

$alignment = foreach ($assetGroup in ($records | Group-Object asset)) {
    $byExactTime = $assetGroup.Group | Group-Object funding_time_unix_ms
    $byMinute = $assetGroup.Group | Group-Object funding_minute_unix_ms
    $counts = @($byMinute | Group-Object Count | Sort-Object { [int]$_.Name } | ForEach-Object { 'venue_count={0}: {1} minute slots' -f $_.Name, $_.Count }) -join '; '
    $threeWay = @($byMinute | Where-Object { $_.Count -eq 3 }).Count
    $anyMinute = $byMinute.Count
    $maxDriftMs = 0
    foreach ($minuteGroup in $byMinute) {
        $stamps = @($minuteGroup.Group | ForEach-Object { [int64]$_.funding_time_unix_ms })
        $drift = ($stamps | Measure-Object -Maximum).Maximum - ($stamps | Measure-Object -Minimum).Minimum
        if ($drift -gt $maxDriftMs) { $maxDriftMs = $drift }
    }
    $exactThreeWay = @($byExactTime | Where-Object { $_.Count -eq 3 }).Count
    $sampleTimes = @($byMinute | Where-Object { $_.Count -eq 3 } | Select-Object -First 3 | ForEach-Object {
        [DateTimeOffset]::FromUnixTimeMilliseconds([int64]$_.Name).ToString('yyyy-MM-ddTHH:mmZ')
    }) -join ', '
    [PSCustomObject]@{
        asset = $assetGroup.Name
        unique_minute_slots = $anyMinute
        minute_slots_shared_by_all_three_venues = $threeWay
        three_way_share_of_minute_slots = if ($anyMinute) { [math]::Round($threeWay / $anyMinute, 4) } else { 0 }
        max_cross_venue_timestamp_drift_ms_within_minute = $maxDriftMs
        exact_millisecond_three_way_matches = $exactThreeWay
        venue_count_distribution = $counts
        first_shared_examples_utc = $sampleTimes
    }
}
$alignment | Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $outDir '第十三周跨所结算时间戳匹配.csv')

Write-Output 'Retrieved rows by venue/asset:'
$venueAsset | Format-Table -AutoSize | Out-String | Write-Output
Write-Output 'Exact timestamp alignment across venues:'
$alignment | Format-Table -AutoSize | Out-String | Write-Output
