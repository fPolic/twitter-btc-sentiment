interface ITicker {
  // Stock symbol
  symbol: string
  // formated: YYYY-MM-DD
  date: string
  // Prices
  open: number
  high: number
  low: number
  close: number
  // Trading volume
  volume: number
}
