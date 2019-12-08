import React from 'react'
import Head from 'next/head'
import {
  XYPlot,
  XAxis,
  YAxis,
  AreaSeries,
  Highlight,
  ChartLabel,
  RadialChart,
  VerticalGridLines,
  HorizontalGridLines,
  LineSeries
} from 'react-vis'
import 'react-vis/dist/style.css'

import _btc from '../public/btc.json'
import _em from '../public/emotions.json'

const emotions = _em.map(d => ({
  ...d,
  date: new Date(d.date.$date)
}))

const DATE_LIMIT = emotions[emotions.length - 1].date

const btc = _btc
  .map(d => ({
    x: new Date(d.date.$date),
    y: d.close
  }))
  .filter(d => d.x < DATE_LIMIT)

// const btc_volume = _btc
//   .map(d => ({
//     x: new Date(d.date.$date),
//     y: d.volume
//     // yo: 280
//   }))
//   .filter(d => d.x < DATE_LIMIT)

const anger = emotions.map(d => ({ x: d.date, y: d.anger }))
const anticipation = emotions.map(d => ({ x: d.date, y: d.anticipation }))

const maxPoints = 1000

const Home = () => {
  const [lastDrawLocation, setLastDrawLocation] = React.useState()

  const filtered = lastDrawLocation
    ? emotions.filter(
        d => d.date > lastDrawLocation.left && d.date < lastDrawLocation.right
      )
    : emotions
  const total_pos = filtered.reduce((acc, d) => acc + d.positive, 0)
  const total_neg = filtered.reduce((acc, d) => acc + d.negative, 0)

  const polarity = [
    {
      angle: total_pos
    },
    {
      angle: total_neg
    }
  ]

  return (
    <div>
      <Head>
        <title>BTC data</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div style={{ display: 'flex' }}>
        <div style={{ flex: '3 1 1000px' }}>
          <XYPlot
            animation
            xDomain={
              lastDrawLocation && [
                lastDrawLocation.left,
                lastDrawLocation.right
              ]
            }
            width={1000}
            height={280}
            xType="time"
          >
            <HorizontalGridLines />
            <VerticalGridLines />
            <ChartLabel
              text="Anger"
              includeMargin={false}
              xPercent={0.5}
              yPercent={0.2}
            />
            <LineSeries data={anger} />
            <XAxis />
            <YAxis />

            <Highlight
              onBrushEnd={setLastDrawLocation}
              onDrag={area => {
                setLastDrawLocation({
                  bottom: lastDrawLocation.bottom + (area.top - area.bottom),
                  left: lastDrawLocation.left - (area.right - area.left),
                  right: lastDrawLocation.right - (area.right - area.left),
                  top: lastDrawLocation.top + (area.top - area.bottom)
                })
              }}
            />
          </XYPlot>

          <XYPlot
            animation
            width={1000}
            xDomain={
              lastDrawLocation && [
                lastDrawLocation.left,
                lastDrawLocation.right
              ]
            }
            height={280}
            xType="time"
          >
            <HorizontalGridLines />
            <VerticalGridLines />
            <ChartLabel
              text="Anticipation"
              includeMargin={false}
              xPercent={0.5}
              yPercent={0.2}
            />
            <LineSeries color="tomato" data={anticipation} />
            <XAxis />
            <YAxis />
            <Highlight
              onBrushEnd={setLastDrawLocation}
              onDrag={area => {
                setLastDrawLocation({
                  bottom: lastDrawLocation.bottom + (area.top - area.bottom),
                  left: lastDrawLocation.left - (area.right - area.left),
                  right: lastDrawLocation.right - (area.right - area.left),
                  top: lastDrawLocation.top + (area.top - area.bottom)
                })
              }}
            />
          </XYPlot>

          <XYPlot
            animation
            width={1000}
            xDomain={
              lastDrawLocation && [
                lastDrawLocation.left,
                lastDrawLocation.right
              ]
            }
            height={280}
            xType="time"
          >
            <HorizontalGridLines />
            <VerticalGridLines />
            <ChartLabel
              text="BTC-USD"
              includeMargin={false}
              xPercent={0.5}
              yPercent={0.2}
            />
            <LineSeries color="gold" data={btc} />
            {/* <AreaSeries data={btc_volume} /> */}
            <XAxis />
            <YAxis />
            <Highlight
              onBrushEnd={setLastDrawLocation}
              onDrag={area => {
                setLastDrawLocation({
                  bottom: lastDrawLocation.bottom + (area.top - area.bottom),
                  left: lastDrawLocation.left - (area.right - area.left),
                  right: lastDrawLocation.right - (area.right - area.left),
                  top: lastDrawLocation.top + (area.top - area.bottom)
                })
              }}
            />
          </XYPlot>
        </div>

        <div
          style={{
            flex: '1 1 500px',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center'
          }}
        >
          <RadialChart animation data={polarity} width={280} height={280} />
        </div>
      </div>

      <style jsx>{``}</style>
    </div>
  )
}

export default Home
