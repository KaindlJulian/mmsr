import axios from 'axios';
import { useEffect, useState } from 'react';
import { Autocomplete, TextField, Grid, Button, CircularProgress } from '@mui/material';
import ResultsTable from './ResultsTable';

export default function App() {
  const [options, setOptions] = useState([]);
  const [loadingOptions, setLoadingOptions] = useState(true);
  const [song, setSong] = useState('');
  const [artist, setArtist] = useState('');
  const [selectedSystem, setSelectedSystem] = useState('ivec256');
  const [similarSongs, setSimilarSongs] = useState(null);

  useEffect(() => {
    axios.get('http://localhost:5000/allSongs')
      .then((res) => {
        setLoadingOptions(false);
        setOptions(JSON.parse(res.data));
      })
      .catch((err) => {
        setLoadingOptions(false);
        console.log(err);
      });
  }, []);

  function startSearch() {
    return () => {
      axios.post('http://localhost:5000/' + selectedSystem, {
        title: song,
        artist: artist
      })
        .then((res) => {
          console.log(JSON.parse(res.data));
          setSimilarSongs(JSON.parse(res.data));
        })
        .catch((err) => {
          console.log(err);
        });
    }
  }

  return (
    <>
      <Grid
        container
        spacing={0}
        direction="column"
        alignItems="center"
      >
        <Grid item xs={3}>
          <h1>Top 10 similar songs</h1>
        </Grid>
        <Grid item xs={3}>
          {loadingOptions ?
            <div>
              <CircularProgress />
            </div>
            :
            <Autocomplete
              disablePortal
              options={options}
              onChange={(e, value) => {
                setSong(value.song);
                setArtist(value.artist);
              }}
              getOptionLabel={(option) => option.song + ' - ' + option.artist}
              sx={{ width: 300, mt: 2 }}
              renderOption={(props, option) => {
                return <li {...props} key={option.id}>
                  {option.song} - {option.artist}
                </li>
              }}
              renderInput={(params) => {
                return <TextField {...params} label="Song" />
              }}
            />
          }
        </Grid>
        <Grid item xs={3}>
          <Button
            variant="contained"
            size="large"
            sx={{ borderRadius: '50px', mt: 2 }}
            onClick={startSearch()}
          >
            Search
          </Button>
        </Grid>
      </Grid>
      <ResultsTable similarSongs={similarSongs}/>
    </>

  );
}
