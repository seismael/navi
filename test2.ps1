$s = @{ i = @() }; try { throw 'test_error' } catch { $s.i += $_.Exception.Message }; $s | ConvertTo-Json -Depth 5
