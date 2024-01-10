import React from 'react';
import { Alert, Grid, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper } from '@mui/material';
import EmbeddedVideo from './EmbeddedVideo';

export default function ResultsTable(props) {
    const similarSongs = props.similarSongs;

    if (similarSongs == null) {
        return (
            <Grid
                container
                spacing={0}
                direction="column"
                alignItems="center"
            >
                <Grid sx={{ mt: 3 }} item xs={3}>
                    <Alert severity="info">Search for a song to get started!</Alert>
                </Grid>
            </Grid>
        )
    } else {
        return (
            <Grid
                container
                spacing={0}
                direction="column"
                alignItems="center"
            >
                <Grid sx={{ mt: 3 }} item xs={3}>
                    <TableContainer component={Paper} sx={{ maxWidth: '70vw' }}>
                        <Table aria-label="results table">
                            <TableHead>
                                <TableRow>
                                    <TableCell>Song</TableCell>
                                    <TableCell>Album</TableCell>
                                    <TableCell>Artist</TableCell>
                                    <TableCell>Genres</TableCell>
                                    <TableCell>Similarity</TableCell>
                                    <TableCell>Youtube</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {similarSongs.map((e) => {
                                    return <TableRow key={e.index}>
                                        <TableCell>{e.song}</TableCell>
                                        <TableCell>{e.album_name}</TableCell>
                                        <TableCell>{e.artist}</TableCell>
                                        <TableCell>{e.genre.reduce((acc, curr) => acc + ', ' + curr)}</TableCell>
                                        <TableCell>{e.similarity}</TableCell>
                                        <TableCell><EmbeddedVideo url={e.url}/></TableCell>
                                    </TableRow>
                                })}
                            </TableBody>
                        </Table>
                    </TableContainer>
                </Grid>
            </Grid>

        );
    }

};
